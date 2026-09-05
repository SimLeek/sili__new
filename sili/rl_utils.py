"""Small reusable RL utilities shared across sili's curiosity/RTAC
experiments. Currently just PopArt; grows as more agents need shared
pieces. See docs/research/rl_utils.rst for the PopArt design rationale.
"""

from __future__ import annotations

import math


class PopArt:
    """PopArt output normalization (van Hasselt et al. 2016; rtrl.nn.PopArt):
    EMA mean/std of the value target, critic trained in normalized space,
    output layer rescaled on every update so original-space predictions are
    invariant to the renormalization itself. See docs/research/rl_utils.rst
    (anchors popart.pop_invariance / popart.derivation /
    popart.generalization)."""

    def __init__(self, beta: float = 0.0003, start_pop: int = 8, eps: float = 1e-6):
        self.beta = beta
        self.mean = 0.0
        self.mean_sq = 0.0
        self.std = 1.0
        self.n = 0
        self.start_pop = start_pop
        self.eps = eps

    def normalize(self, raw: float) -> float:
        """Normalize a raw value using CURRENT statistics (no update)."""
        return (raw - self.mean) / max(self.std, self.eps)

    def unnormalize(self, normalized: float) -> float:
        return normalized * self.std + self.mean

    def update_and_rescale(self, raw_target: float, weight_arrays: list, bias_arrays: list) -> float:
        """Update running mean/std with raw_target (EMA), then rescale
        weight_arrays (scale only -- arrays that multiply input features)
        and bias_arrays (scale + shift -- arrays that add independent of
        input) in place per docs/research/rl_utils.rst:popart.derivation.
        Returns the normalized target to train the critic against.
        """
        self.n += 1
        old_mean, old_std = self.mean, self.std

        self.mean = (1 - self.beta) * self.mean + self.beta * raw_target
        self.mean_sq = (1 - self.beta) * self.mean_sq + self.beta * (raw_target**2)
        var = max(self.mean_sq - self.mean**2, self.eps)
        self.std = math.sqrt(var)

        if self.n > self.start_pop:
            # See docs/research/rl_utils.rst:popart.start_pop_guard.
            scale = old_std / max(self.std, self.eps)
            shift = (old_mean - self.mean) / max(self.std, self.eps)
            for w in weight_arrays:
                w *= scale
            for b in bias_arrays:
                b *= scale
                b += shift

        return self.normalize(raw_target)
