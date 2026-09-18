#pragma once

// Real-time engine selection: dense/sparse give IDENTICAL results within
// a group, only speed differs. Cross-validated decision trees fit on
// width, batch, density, and (Group A) synapse density -- see docs/
// research/sparse_rnn.rst:sparse_rnn.engine_select_two_groups for the
// full derivation, measured accuracy, and known gaps (not repeated
// here), and scripts/fit_engine_select_rules.py to regenerate.

// Group A (disldo/sisldo forward). syn_density unused -- the tree never
// selected it as a useful split for forward (only backward, below).
inline bool group_a_forward_use_sisldo(float n, int batch, float density) {
    if (density <= 0.158114f) {
        if (density <= 0.015811f) {
            return true;
        } else {
            if (batch <= 45.2548f) {
                return true;
            } else {
                if (n <= 192.0000f) {
                    return true;
                } else {
                    if (n <= 1448.1547f) {
                        return false;
                    } else {
                        return n <= 2896.3094f;
                    }
                }
            }
        }
    } else {
        return batch <= 5.6569f;
    }
}

// Group A (disldo/sisldo backward). syn_density = nnz / (n_inputs *
// n_outputs) -- the WEIGHT matrix's own sparsity, a property of the layer,
// not of this call's activations/gradients. A real, previously-uncaptured
// driver: at density > ~0.16, sisldo stays competitive far longer when the
// weight matrix itself is sparse (syn_density <= ~0.32) than when it's
// dense, because dense-weight disldo's inner loop has nothing to skip.
inline bool group_a_backward_use_sisldo(float n, int batch, float density, float syn_density) {
    if (density <= 0.158114f) {
        return true;
    } else {
        if (syn_density <= 0.316228f) {
            if (n <= 1448.1547f) {
                if (n <= 384.0000f) {
                    if (n <= 192.0000f) {
                        return true;
                    } else {
                        return batch <= 45.2548f;
                    }
                } else {
                    return true;
                }
            } else {
                return batch <= 5.6569f;
            }
        } else {
            return batch <= 5.6569f;
        }
    }
}

// Group B (didldo/sidldo forward).
inline bool group_b_forward_use_sidldo(float n, int batch, float density) {
    if (density <= 0.015811f) {
        return n > 90.5097f;
    } else {
        if (batch <= 5.6569f) {
            return density <= 0.316228f;
        } else {
            return false;
        }
    }
}

// Group B (didldo/sidldo backward) -- independently fit from real bwd0/
// bwdX data across all 7 widths (previously only n=288 was ever tested).
inline bool group_b_backward_use_sidldo(float n, int batch, float density) {
    if (batch <= 5.6569f) {
        return true;
    } else {
        if (density <= 0.015811f) {
            if (n <= 192.0000f) {
                return batch <= 90.5097f;
            } else {
                return true;
            }
        } else {
            if (n <= 2896.3094f) {
                return false;
            } else {
                return density <= 0.070711f;
            }
        }
    }
}
