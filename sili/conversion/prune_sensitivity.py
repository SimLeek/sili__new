"""
prune_sensitivity.py
─────────────────────
Per-tensor-role pruning sensitivity: does this model tolerate the SAME
sparsity threshold on every 2-D matrix, or do different weight roles
(attention Q vs V, MLP gate vs down, embeddings, ...) need very
different thresholds to avoid destroying the model?

Motivating finding (see sili_peridot's own conversion notes): on
MiniCPM5-1B-Base, embed_tokens tolerated 90% magnitude-pruned sparsity
with only a mild quality drop, while v_proj (architecturally similar to
k_proj, which tolerated 70% fine) collapsed already at 30%. A single
global calibrate_min_abs_param() threshold cannot serve both of those
tensors well -- one group's safe ceiling is another group's already
-catastrophic floor. This module makes that kind of per-role analysis a
reusable procedure instead of one-off manual sweeps, for whichever
downstream task (next-token perplexity, classification accuracy, a
custom metric) the caller cares about -- `eval_fn` is the only
model/task-specific piece; everything else here is generic.

Not part of the sili runtime path -- this is a conversion-time
diagnostic/calibration tool, like sparse_prune.py's calibration
functions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from .rnn_fold import _parse_block_key
from .sparse_prune import calibrate_min_abs_param

EvalFn = Callable[[dict[str, torch.Tensor]], float]


def group_tensor_names_by_role(state_dict: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    """
    Group 2-D tensor names by "role": tensors that recur once per
    repeated block (e.g. every layer's own self_attn.q_proj.weight) are
    grouped by their shared (prefix, suffix) -- matching
    rnn_fold.detect_repeated_block_groups' own notion of a repeated
    block, since these are the tensors a single fold-time CSR threshold
    will eventually apply to uniformly. Tensors with no numeric block
    index (embed_tokens, lm_head, a final norm, ...) each get their own
    singleton group under their full name -- there's nothing to group
    them WITH.

    Only 2-D tensors are included (matches _keep_dense_reason /
    calibrate_min_abs_param's own eligibility rule -- vectors and higher
    -rank tensors are never threshold-pruned, so grouping them for a
    sensitivity sweep would be meaningless).
    """
    groups: dict[str, list[str]] = {}
    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor) or t.ndim != 2:
            continue
        parsed = _parse_block_key(name)
        key = f"*{parsed[0]}{parsed[2]}" if parsed is not None else name
        groups.setdefault(key, []).append(name)
    return groups


def sweep_group_sensitivity(
    state_dict: dict[str, torch.Tensor],
    groups: dict[str, list[str]],
    eval_fn: EvalFn,
    sparsity_grid: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8),
    max_sample: int = 5_000_000,
) -> dict[str, dict[float, float]]:
    """
    For each group, sweep `sparsity_grid` pruning ONLY that group's
    tensors (every other tensor stays exactly as given in `state_dict`),
    and record eval_fn's score at each point. Returns {group_name:
    {sparsity: score}}.

    eval_fn takes a full {name: tensor} state dict (same keys as
    `state_dict`, with one group's tensors possibly zeroed below a
    threshold) and returns a scalar score where HIGHER IS BETTER (e.g.
    accuracy, -perplexity, -loss) -- this module doesn't care which
    metric, only that comparisons between sparsity levels stay
    meaningful. This is the one genuinely expensive part (typically a
    full model forward pass) -- everything else here is bookkeeping.

    Cost: len(groups) * len(sparsity_grid) calls to eval_fn. For a large
    model with many groups, budget accordingly (see
    examples/prune_sensitivity_example.py for a worked, timed example).
    """
    results: dict[str, dict[float, float]] = {}
    for gname, names in groups.items():
        sub_sd = {n: state_dict[n] for n in names}
        per_target: dict[float, float] = {}
        for target in sparsity_grid:
            threshold = (
                calibrate_min_abs_param(sub_sd, target_sparsity=target, max_sample=max_sample) if target > 0.0 else 0.0
            )
            trial_sd = dict(state_dict)
            for n in names:
                t = state_dict[n].detach().float()
                trial_sd[n] = t * (t.abs() >= threshold)
            per_target[target] = eval_fn(trial_sd)
        results[gname] = per_target
    return results


def apply_group_thresholds(
    state_dict: dict[str, torch.Tensor],
    groups: dict[str, list[str]],
    target_sparsity_by_group: dict[str, float],
    max_sample: int = 5_000_000,
) -> dict[str, torch.Tensor]:
    """
    Prune each group named in `target_sparsity_by_group` at its OWN
    calibrated threshold (calibrated on that group's tensors only, not
    the whole model); groups not present in `target_sparsity_by_group`
    are left untouched. Returns a new dense {name: tensor} dict (no CSR
    conversion -- that's sparse_prune.to_sparse_csr's job, once a
    per-group decision has actually been made and verified).
    """
    out = dict(state_dict)
    for gname, target in target_sparsity_by_group.items():
        names = groups[gname]
        sub_sd = {n: state_dict[n] for n in names}
        threshold = (
            calibrate_min_abs_param(sub_sd, target_sparsity=target, max_sample=max_sample) if target > 0.0 else 0.0
        )
        for n in names:
            t = state_dict[n].detach().float()
            out[n] = t * (t.abs() >= threshold)
    return out


def stepwise_cumulative_eval(
    state_dict: dict[str, torch.Tensor],
    groups: dict[str, list[str]],
    target_sparsity_by_group: dict[str, float],
    step_order: Sequence[Sequence[str]],
    eval_fn: EvalFn,
    max_sample: int = 5_000_000,
) -> list[tuple[tuple[str, ...], float]]:
    """
    Add groups CUMULATIVELY, one step at a time (each step can add more
    than one group at once -- e.g. a first step of ["q_proj", "k_proj"]
    together), pruning every group added so far (at its own threshold
    from `target_sparsity_by_group`) while every group not yet reached
    stays at its ORIGINAL (dense) values. Evaluates against `state_dict`
    itself as the baseline reference at every step (not against the
    previous step's result) -- the point is to see whether combining
    per-group thresholds compounds into a worse-than-expected quality
    loss, which isolated single-group sweeps (sweep_group_sensitivity)
    can't reveal on their own.

    Returns a list of (groups_added_so_far, score), in step_order.
    """
    cumulative: list[str] = []
    trial_sd = dict(state_dict)
    out: list[tuple[tuple[str, ...], float]] = []

    for step in step_order:
        for gname in step:
            cumulative.append(gname)
            names = groups[gname]
            sub_sd = {n: state_dict[n] for n in names}
            target = target_sparsity_by_group[gname]
            threshold = (
                calibrate_min_abs_param(sub_sd, target_sparsity=target, max_sample=max_sample) if target > 0.0 else 0.0
            )
            for n in names:
                t = state_dict[n].detach().float()
                trial_sd[n] = t * (t.abs() >= threshold)
        out.append((tuple(cumulative), eval_fn(trial_sd)))
    return out


def iterative_threshold_search(
    state_dict: dict[str, torch.Tensor],
    groups: dict[str, list[str]],
    initial_thresholds: dict[str, float],
    step_order: Sequence[Sequence[str]],
    eval_fn: EvalFn,
    baseline_score: float,
    target_score: float,
    shrink_factor: float = 0.5,
    max_iterations: int = 10,
    min_threshold: float = 0.0,
    max_sample: int = 5_000_000,
) -> tuple[dict[str, float], list[dict]]:
    """
    Greedy search: run stepwise_cumulative_eval with the CURRENT
    thresholds, find whichever step caused the single worst marginal
    score drop, shrink JUST the group(s) newly added in that step by
    `shrink_factor` (floored at `min_threshold`), and repeat. Stops as
    soon as the final (all-groups-combined) score reaches
    `target_score`, or after `max_iterations` rounds, or once nothing is
    left with room to shrink -- whichever comes first.

    This encodes the "look at the stepwise trace, reduce whatever caused
    the biggest single jump, re-check the combined result" procedure
    used to calibrate MiniCPM5-1B-Base's own per-group thresholds (see
    sili_peridot's conversion notes) -- greedy and NOT guaranteed
    globally optimal (shrinking one group changes how much a group added
    AFTER it costs, since costs compound along step_order; only the
    single worst offender per round gets touched, never a joint
    optimization across all groups at once), but it's exactly the manual
    procedure that worked there, made repeatable instead of ad hoc.

    baseline_score: eval_fn's score on the fully dense `state_dict` --
    passed in rather than recomputed each round, since eval_fn is
    typically a full model forward pass and this value never changes.

    Returns (final_thresholds, history) -- history has one entry per
    iteration actually run: {"thresholds": {...}, "final_score": float,
    "steps": [(names_so_far, score), ...]}, in order, so callers can see
    the whole search trace, not just the final answer.
    """
    thresholds = dict(initial_thresholds)
    history: list[dict] = []

    for _ in range(max_iterations):
        steps = stepwise_cumulative_eval(state_dict, groups, thresholds, step_order, eval_fn, max_sample=max_sample)
        final_score = steps[-1][1]
        history.append({"thresholds": dict(thresholds), "final_score": final_score, "steps": steps})
        if final_score >= target_score:
            break

        drops = []
        prev_score = baseline_score
        for i, (_names, score) in enumerate(steps):
            drops.append((prev_score - score, i))
            prev_score = score
        drops.sort(key=lambda d: -d[0])

        shrunk_any = False
        for _drop, step_idx in drops:
            names = steps[step_idx][0]
            prev_names = steps[step_idx - 1][0] if step_idx > 0 else ()
            newly_added = [n for n in names if n not in prev_names]
            shrinkable = [g for g in newly_added if thresholds[g] > min_threshold]
            if shrinkable:
                for gname in shrinkable:
                    thresholds[gname] = max(min_threshold, thresholds[gname] * shrink_factor)
                shrunk_any = True
                break
        if not shrunk_any:
            break  # nothing left with room to shrink -- stop rather than loop forever

    return thresholds, history
