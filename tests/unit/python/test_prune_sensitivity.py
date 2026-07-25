"""
tests/unit/python/test_prune_sensitivity.py
─────────────────────────────────────────────
prune_sensitivity.py's bookkeeping (grouping, sweeping, applying
per-group thresholds, cumulative stepwise eval) is pure logic around an
injected eval_fn -- tested here with small synthetic state dicts and
closed-form eval_fn's (no real model, no torch.nn), not the real
next-token-prediction eval_fn sili_peridot actually uses for MiniCPM5.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
import pytest

from sili.conversion.prune_sensitivity import (
    group_tensor_names_by_role, sweep_group_sensitivity,
    apply_group_thresholds, stepwise_cumulative_eval,
)


def _toy_state_dict(n_layers=3):
    sd = {
        "model.embed_tokens.weight": torch.randn(20, 8),
        "lm_head.weight":            torch.randn(20, 8),
        "model.norm.weight":         torch.randn(8),   # 1-D -- excluded
    }
    for i in range(n_layers):
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(8, 8) + 1.0
        # Distinct increasing values (not uniform): a uniform tensor makes
        # target_sparsity=1.0's threshold equal every element's value, and
        # the mask is `>=`, so nothing would ever actually get pruned.
        sd[f"model.layers.{i}.self_attn.v_proj.weight"] = (
            torch.arange(1, 65, dtype=torch.float32).reshape(8, 8) * 0.1)
        sd[f"model.layers.{i}.input_layernorm.weight"]  = torch.randn(8)  # 1-D -- excluded
    return sd


class TestGroupTensorNamesByRole:
    def test_repeated_block_tensors_share_a_group(self):
        sd = _toy_state_dict(n_layers=3)
        groups = group_tensor_names_by_role(sd)
        q_group = [g for g in groups if "q_proj" in g]
        assert len(q_group) == 1
        assert len(groups[q_group[0]]) == 3

    def test_singleton_tensors_get_their_own_group(self):
        sd = _toy_state_dict(n_layers=3)
        groups = group_tensor_names_by_role(sd)
        assert groups["model.embed_tokens.weight"] == ["model.embed_tokens.weight"]
        assert groups["lm_head.weight"] == ["lm_head.weight"]

    def test_1d_tensors_excluded(self):
        sd = _toy_state_dict(n_layers=3)
        groups = group_tensor_names_by_role(sd)
        all_names = {n for names in groups.values() for n in names}
        assert "model.norm.weight" not in all_names
        assert "model.layers.0.input_layernorm.weight" not in all_names

    def test_different_prefixes_dont_merge(self):
        sd = {
            "model.language_model.layers.0.mlp.weight": torch.randn(4, 4),
            "model.language_model.layers.1.mlp.weight": torch.randn(4, 4),
            "model.vision_tower.layers.0.mlp.weight":    torch.randn(4, 4),
        }
        groups = group_tensor_names_by_role(sd)
        mlp_groups = [g for g in groups if "mlp" in g]
        assert len(mlp_groups) == 2   # language and vision stay separate


class TestSweepGroupSensitivity:
    def test_only_the_swept_groups_own_tensor_affects_its_score(self):
        sd = _toy_state_dict(n_layers=2)
        groups = group_tensor_names_by_role(sd)

        # eval_fn only cares about how many nonzeros remain in v_proj's
        # tensors -- q_proj pruning should never move this score.
        v_names = groups[[g for g in groups if "v_proj" in g][0]]

        def eval_fn(trial_sd):
            return float(sum((trial_sd[n] != 0).sum().item() for n in v_names))

        results = sweep_group_sensitivity(sd, groups, eval_fn,
                                          sparsity_grid=(0.0, 0.5, 1.0))
        q_group = [g for g in groups if "q_proj" in g][0]
        v_group = [g for g in groups if "v_proj" in g][0]

        # Pruning q_proj (a different group) must not change v_proj's
        # nonzero count -- score stays exactly the same across the grid.
        assert len(set(results[q_group].values())) == 1
        # Pruning v_proj itself: target=1.0's threshold is the max value,
        # so the mask (>=) keeps only the single largest element per
        # tensor -- nearly everything else is zeroed.
        assert results[v_group][1.0] < results[v_group][0.0] * 0.1

    def test_sparsity_grid_zero_leaves_score_at_baseline(self):
        sd = _toy_state_dict(n_layers=2)
        groups = group_tensor_names_by_role(sd)

        def eval_fn(trial_sd):
            return float(sum((v != 0).sum().item() for v in trial_sd.values()))

        baseline = eval_fn(sd)
        results = sweep_group_sensitivity(sd, groups, eval_fn, sparsity_grid=(0.0,))
        for gname in groups:
            assert results[gname][0.0] == pytest.approx(baseline)


class TestApplyGroupThresholds:
    def test_only_named_groups_get_pruned(self):
        sd = _toy_state_dict(n_layers=2)
        groups = group_tensor_names_by_role(sd)
        v_group = [g for g in groups if "v_proj" in g][0]

        out = apply_group_thresholds(sd, groups, {v_group: 1.0})
        for n in groups[v_group]:
            # target=1.0's threshold is the tensor's own max -- only that
            # single largest element survives the >= mask.
            assert int((out[n] != 0).sum()) == 1
        # untouched groups keep their original values exactly.
        q_group = [g for g in groups if "q_proj" in g][0]
        for n in groups[q_group]:
            assert torch.equal(out[n], sd[n])

    def test_target_zero_is_a_no_op(self):
        sd = _toy_state_dict(n_layers=2)
        groups = group_tensor_names_by_role(sd)
        v_group = [g for g in groups if "v_proj" in g][0]
        out = apply_group_thresholds(sd, groups, {v_group: 0.0})
        for n in groups[v_group]:
            assert torch.equal(out[n], sd[n])


class TestStepwiseCumulativeEval:
    def test_cumulative_not_independent(self):
        sd = _toy_state_dict(n_layers=2)
        groups = group_tensor_names_by_role(sd)
        q_group = [g for g in groups if "q_proj" in g][0]
        v_group = [g for g in groups if "v_proj" in g][0]

        def eval_fn(trial_sd):
            return float(sum((v != 0).sum().item() for v in trial_sd.values()))

        baseline = eval_fn(sd)
        steps = stepwise_cumulative_eval(
            sd, groups,
            target_sparsity_by_group={q_group: 1.0, v_group: 1.0},
            step_order=[[q_group], [v_group]],
            eval_fn=eval_fn,
        )
        (names1, score1), (names2, score2) = steps
        assert names1 == (q_group,)
        assert names2 == (q_group, v_group)
        # Each step prunes strictly more than the last -- score keeps dropping.
        assert score1 < baseline
        assert score2 < score1

    def test_multi_group_single_step(self):
        sd = _toy_state_dict(n_layers=2)
        groups = group_tensor_names_by_role(sd)
        q_group = [g for g in groups if "q_proj" in g][0]
        v_group = [g for g in groups if "v_proj" in g][0]

        def eval_fn(trial_sd):
            return float(sum((v != 0).sum().item() for v in trial_sd.values()))

        steps = stepwise_cumulative_eval(
            sd, groups,
            target_sparsity_by_group={q_group: 1.0, v_group: 1.0},
            step_order=[[q_group, v_group]],   # both added together
            eval_fn=eval_fn,
        )
        assert len(steps) == 1
        assert steps[0][0] == (q_group, v_group)
