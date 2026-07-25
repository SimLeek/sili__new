"""
Worked example: sili.conversion.prune_sensitivity, the per-tensor-role
pruning sensitivity workflow.

Background (see sili_peridot's own conversion notes for the full story):
converting MiniCPM5-1B-Base, a single global calibrate_min_abs_param()
threshold turned out to be the wrong tool entirely -- embed_tokens
tolerated 90% magnitude-pruned sparsity with only a mild quality drop,
while v_proj (architecturally similar to k_proj, which tolerated 70%
fine) collapsed already past ~25% sparsity. One group's safe ceiling was
another group's already-catastrophic floor. This module makes that kind
of per-role analysis (which this file demonstrates end-to-end on a toy
model) a reusable procedure for the NEXT model, instead of a one-off
manual investigation.

The only genuinely model-specific piece is `eval_fn`: something that
takes a candidate state dict and returns a scalar score where HIGHER IS
BETTER. For a causal LM, that's usually next-token perplexity/accuracy
via teacher forcing on a small held-out text sample (loading the real
HuggingFace model twice -- once dense, once with a candidate group
pruned -- exactly as the docstring below shows, just using a toy linear
model here instead of a real 1B-parameter checkpoint so this file runs
in under a second with no external downloads).

Run: python -m examples.prune_sensitivity_example
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from sili.conversion.prune_sensitivity import (
    group_tensor_names_by_role, sweep_group_sensitivity,
    apply_group_thresholds, stepwise_cumulative_eval,
)


# ── Toy target: a tiny "3-layer" model where one layer's weight is
#    deliberately much more sensitive to pruning than the others (in a
#    real model this asymmetry is discovered, not injected -- see the
#    real MiniCPM5 numbers in the module docstring above).

def make_toy_state_dict():
    torch.manual_seed(0)
    sd = {}
    for i in range(3):
        # "mlp": ordinary small-magnitude values -- losing the smallest
        # fraction costs eval_fn (an L1 distance to the original) little.
        sd[f"model.layers.{i}.mlp.weight"] = torch.randn(16, 16)
        # "critical": dense, no near-zero entries, ~10x larger magnitude
        # -- the SAME pruned *fraction* costs eval_fn ~10x more here,
        # standing in for a role that's disproportionately sensitive to
        # magnitude pruning (e.g. v_proj in the real MiniCPM5 numbers
        # this module's docstring references).
        sd[f"model.layers.{i}.attn.critical.weight"] = torch.randn(16, 16) * 2.0 + 10.0
    return sd


def make_eval_fn(reference_sd):
    """Toy stand-in for "run the real model and measure task quality":
    higher score = state dict closer to the reference. A real eval_fn
    would tokenize a held-out text sample, run the HF model with
    labels=input_ids, and return something like -loss or accuracy --
    see sili_peridot's model/eval_pruning.py for that actual
    implementation (not duplicated here since it needs transformers +
    the real checkpoint, neither of which this example depends on)."""
    def eval_fn(trial_sd):
        return -sum(float((trial_sd[k] - reference_sd[k]).abs().sum())
                   for k in reference_sd)
    return eval_fn


def main():
    sd = make_toy_state_dict()
    eval_fn = make_eval_fn(sd)
    groups = group_tensor_names_by_role(sd)
    print("Detected groups:", list(groups.keys()))

    # Step 1: sweep each group in isolation to see its own sensitivity.
    sensitivity = sweep_group_sensitivity(
        sd, groups, eval_fn, sparsity_grid=(0.0, 0.3, 0.6, 0.9))
    print("\nPer-group sensitivity (score at each target_sparsity):")
    for gname, scores in sensitivity.items():
        print(f"  {gname:35s} " + " ".join(f"{t}:{s:8.2f}" for t, s in scores.items()))

    # Step 2: pick per-group thresholds from that table (a judgment call
    # -- see sili_peridot's own notes on why this stayed a human decision
    # rather than an automatic policy). Here: the "critical" role is
    # fragile, so it keeps a low target; the ordinary mlp weights can
    # take more.
    mlp_group    = next(g for g in groups if "mlp" in g)
    crit_group   = next(g for g in groups if "critical" in g)
    thresholds   = {mlp_group: 0.6, crit_group: 0.1}

    # Step 3: verify the COMBINED effect stays reasonable -- add groups
    # cumulatively (not all-at-once) so a compounding interaction shows
    # up as a step-over-step score drop, evaluated against the ORIGINAL
    # dense state dict at every step (not the previous step's result).
    print("\nStepwise cumulative eval (score vs. the original dense state dict):")
    steps = stepwise_cumulative_eval(
        sd, groups, thresholds,
        step_order=[[mlp_group], [crit_group]],
        eval_fn=eval_fn,
    )
    for names_so_far, score in steps:
        print(f"  after {names_so_far}: score={score:.2f}")

    final = apply_group_thresholds(sd, groups, thresholds)
    print("\nFinal pruned state dict tensor names:", list(final.keys()))


if __name__ == "__main__":
    main()
