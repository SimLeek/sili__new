#pragma once

// Real-time engine selection: dense/sparse give IDENTICAL results within
// a group, only speed differs. Coarse first cuts -- see docs/research/
// sparse_rnn.rst:sparse_rnn.engine_select_two_groups for derivation,
// sample sizes, and known gaps (not repeated here).

// Group A (disldo/sisldo forward).
inline bool group_a_forward_use_sisldo(int batch, float density) {
    if (batch == 1)
        return true;
    return density < 0.05f;
}

// Group A (disldo/sisldo backward) -- known gap, less clean than forward.
inline bool group_a_backward_use_sisldo(int batch, float density) {
    if (batch == 1)
        return true;
    return density < 0.05f;
}

// Group B (didldo/sidldo forward) -- two-tier, crossover shifts ~10x.
inline bool group_b_forward_use_sidldo(int batch, float density) {
    const float threshold = (batch <= 1) ? 0.1f : 0.02f;
    return density < threshold;
}

// Group B (didldo/sidldo backward) -- UNVALIDATED placeholder.
inline bool group_b_backward_use_sidldo(int batch, float density) {
    return group_b_forward_use_sidldo(batch, density);
}
