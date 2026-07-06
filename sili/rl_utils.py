"""
rl_utils.py -- small reusable RL utilities shared across sili's curiosity/RTAC
experiments. Currently just PopArt; grows as more agents need shared pieces.
"""
from __future__ import annotations
import math
import numpy as np


class PopArt:
    """
    PopArt output normalization (van Hasselt et al. 2016; used in rtrl/rtac.py
    via rtrl.nn.PopArt). Learns a running mean/std of the (potentially
    nonstationary, unknown-scale) value target via EMA, and trains the critic
    against the NORMALIZED target so gradient magnitudes stay well-behaved
    regardless of the raw reward scale.

    The "Pop" half is the part that's easy to get wrong: whenever mean/std
    update, the output layer's weights must be rescaled so PREDICTIONS IN THE
    ORIGINAL (unnormalized) SPACE are unchanged by the renormalization event
    itself -- only by actual learning. Without this, every EMA update of
    mean/std silently perturbs every existing prediction, which fights the
    critic's own gradient updates.

    Derivation for a linear value head v(h,a) = h @ W + b(a) (here b(a) is
    itself a per-action row of a "bias" matrix, since our value head is
    conditioned on the one-hot previous action rather than a single additive
    scalar bias -- see test_mandelbrot_rl.py's Wv_h / Wv_a):

        normalized_pred = h @ W + b
        original_pred    = normalized_pred * std_old + mean_old
        We want original_pred to be reproduced by the NEW normalization:
            original_pred = new_normalized_pred * std_new + mean_new
        Solving for new_normalized_pred:
            new_normalized_pred = (normalized_pred*std_old + mean_old - mean_new) / std_new
                                 = h @ (W * std_old/std_new)
                                   + (b*std_old + mean_old - mean_new) / std_new
        So: W_new = W_old * scale            (weight-like arrays: SCALE ONLY)
            b_new = b_old * scale + shift     (bias-like arrays: SCALE + SHIFT)
        where scale = std_old/std_new, shift = (mean_old - mean_new)/std_new.

    This generalizes beyond a single scalar bias to any array that acts as an
    additive term in the pre-normalization prediction (e.g. every row of a
    one-hot-selected bias matrix gets the same shift, since exactly one row
    is ever active per prediction).
    """

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

    def update_and_rescale(self, raw_target: float,
                           weight_arrays: list, bias_arrays: list) -> float:
        """
        Update running mean/std with raw_target (EMA), then rescale the given
        weight-like arrays (scale only) and bias-like arrays (scale + shift)
        IN PLACE so existing predictions are unchanged by this update alone.

        weight_arrays : arrays that multiply the (unchanged) input features
                        directly (e.g. the h @ W term)
        bias_arrays   : arrays that act as an additive offset independent of
                        the input (e.g. a per-action bias row)

        Returns the normalized target to train the critic against.
        """
        self.n += 1
        old_mean, old_std = self.mean, self.std

        self.mean    = (1 - self.beta) * self.mean    + self.beta * raw_target
        self.mean_sq = (1 - self.beta) * self.mean_sq + self.beta * (raw_target ** 2)
        var = max(self.mean_sq - self.mean ** 2, self.eps)
        self.std = math.sqrt(var)

        if self.n > self.start_pop:
            # Enough data has accumulated for stable statistics -- perform
            # the Pop rescale. Below start_pop, mean/std still update every
            # call (so statistics warm up from the first sample), but the
            # weight rescale is skipped since scale/shift computed from
            # noisy early statistics would perturb the weights more than
            # they should move (mirrors rtrl.nn.PopArt's start_pop guard).
            scale = old_std / max(self.std, self.eps)
            shift = (old_mean - self.mean) / max(self.std, self.eps)
            for w in weight_arrays:
                w *= scale
            for b in bias_arrays:
                b *= scale
                b += shift

        return self.normalize(raw_target)
