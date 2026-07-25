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

from typing import Callable, Dict, List, Sequence, Tuple

import torch

from .rnn_fold import _parse_block_key
from .sparse_prune import calibrate_min_abs_param

EvalFn = Callable[[Dict[str, torch.Tensor]], float]


def group_tensor_names_by_role(state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
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
    groups: Dict[str, List[str]] = {}
    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor) or t.ndim != 2:
            continue
        parsed = _parse_block_key(name)
        key = f"*{parsed[0]}{parsed[2]}" if parsed is not None else name
        groups.setdefault(key, []).append(name)
    return groups


def sweep_group_sensitivity(
    state_dict: Dict[str, torch.Tensor],
    groups: Dict[str, List[str]],
    eval_fn: EvalFn,
    sparsity_grid: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8),
    max_sample: int = 5_000_000,
) -> Dict[str, Dict[float, float]]:
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
    results: Dict[str, Dict[float, float]] = {}
    for gname, names in groups.items():
        sub_sd = {n: state_dict[n] for n in names}
        per_target: Dict[float, float] = {}
        for target in sparsity_grid:
            threshold = (calibrate_min_abs_param(sub_sd, target_sparsity=target,
                                                 max_sample=max_sample)
                        if target > 0.0 else 0.0)
            trial_sd = dict(state_dict)
            for n in names:
                t = state_dict[n].detach().float()
                trial_sd[n] = t * (t.abs() >= threshold)
            per_target[target] = eval_fn(trial_sd)
        results[gname] = per_target
    return results


def apply_group_thresholds(
    state_dict: Dict[str, torch.Tensor],
    groups: Dict[str, List[str]],
    target_sparsity_by_group: Dict[str, float],
    max_sample: int = 5_000_000,
) -> Dict[str, torch.Tensor]:
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
        threshold = (calibrate_min_abs_param(sub_sd, target_sparsity=target,
                                             max_sample=max_sample)
                    if target > 0.0 else 0.0)
        for n in names:
            t = state_dict[n].detach().float()
            out[n] = t * (t.abs() >= threshold)
    return out


def stepwise_cumulative_eval(
    state_dict: Dict[str, torch.Tensor],
    groups: Dict[str, List[str]],
    target_sparsity_by_group: Dict[str, float],
    step_order: Sequence[Sequence[str]],
    eval_fn: EvalFn,
    max_sample: int = 5_000_000,
) -> List[Tuple[Tuple[str, ...], float]]:
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
    cumulative: List[str] = []
    trial_sd = dict(state_dict)
    out: List[Tuple[Tuple[str, ...], float]] = []

    for step in step_order:
        for gname in step:
            cumulative.append(gname)
            names = groups[gname]
            sub_sd = {n: state_dict[n] for n in names}
            target = target_sparsity_by_group[gname]
            threshold = (calibrate_min_abs_param(sub_sd, target_sparsity=target,
                                                 max_sample=max_sample)
                        if target > 0.0 else 0.0)
            for n in names:
                t = state_dict[n].detach().float()
                trial_sd[n] = t * (t.abs() >= threshold)
        out.append((tuple(cumulative), eval_fn(trial_sd)))
    return out
