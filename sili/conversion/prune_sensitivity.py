"""Per-tensor-role pruning sensitivity calibration. Not part of the sili
runtime path -- a conversion-time diagnostic tool, like sparse_prune.py's
calibration functions. See docs/research/prune_sensitivity.rst for the
motivating MiniCPM5-1B-Base finding
(``prune_sensitivity.motivating_finding``) that different weight roles
need very different pruning thresholds. ``eval_fn`` is the only
model/task-specific piece throughout this module; everything else is
generic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from .rnn_fold import _parse_block_key
from .sparse_prune import calibrate_min_abs_param

EvalFn = Callable[[dict[str, torch.Tensor]], float]


def group_tensor_names_by_role(state_dict: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    """Group 2-D tensor names by "role" (shared prefix/suffix across
    repeated blocks, or a singleton for non-block tensors like
    embed_tokens). See docs/research/prune_sensitivity.rst:
    prune_sensitivity.role_grouping_rule for the full rationale."""
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
    """For each group, sweep `sparsity_grid` pruning ONLY that group's
    tensors (every other tensor stays as given in `state_dict`), and
    record eval_fn's score at each point. Returns {group_name:
    {sparsity: score}}. eval_fn returns a scalar where HIGHER IS BETTER;
    it's the one genuinely expensive part (typically a full model forward
    pass). Cost: len(groups) * len(sparsity_grid) calls to eval_fn (see
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
    """Greedy search: find whichever step caused the single worst marginal
    score drop, shrink JUST that step's newly-added group(s) by
    `shrink_factor` (floored at `min_threshold`), repeat until
    `target_score` is reached, `max_iterations` is hit, or nothing is
    left to shrink. NOT globally optimal (see
    docs/research/prune_sensitivity.rst:prune_sensitivity.iterative_search_greedy
    for why, and the MiniCPM5-1B-Base procedure this encodes).

    baseline_score: eval_fn's score on the fully dense `state_dict`,
    passed in rather than recomputed each round.

    Returns (final_thresholds, history), history having one entry per
    iteration: {"thresholds": {...}, "final_score": float, "steps":
    [(names_so_far, score), ...]}.
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
