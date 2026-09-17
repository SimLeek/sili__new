#!/usr/bin/env python3
"""Fits the dispatch rules in sili/lib/headers/engine_select.hpp from
measured benchmark data (engine_select_bench_data.json, alongside this
script). Not a test -- a generator: re-run after gathering new benchmark
rows (more widths, more syn_density points, a new machine) to refit and
reprint the header content.

Hand-rolled CART (no sklearn available in this environment), features
log2(width), log2(batch), log10(density), and for Group A log10(syn_density)
-- these axes are naturally multiplicative/geometric. Hyperparameters
(max_depth, min_leaf, min_gain) are chosen per rule by k-fold cross-
validation, not training accuracy: an earlier min_leaf=3 pass produced
leaves with 4-8 samples and 50-75% leaf accuracy -- fitting measurement
noise, not real crossover structure -- while reporting inflated training
accuracy. After fitting, `prune_redundant` collapses any split whose two
children both predict the same class (zero accuracy cost -- unlike
rounding/simplifying thresholds for legibility, which does cost accuracy
and was tried and rejected in favor of this exact-threshold approach).
"""

import json
import math
import os
import random

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_select_bench_data.json")


def load_rows():
    with open(DATA_PATH) as f:
        data = json.load(f)
    return data["group_a"], data["group_b"]


class Node:
    __slots__ = ("acc", "feat", "left", "n", "pred", "right", "thresh")

    def __init__(self):
        self.feat = None
        self.thresh = None
        self.left = None
        self.right = None
        self.pred = None
        self.n = 0
        self.acc = None


def featurize_a(r):
    return {
        "logn": math.log2(r["n"]),
        "logbatch": math.log2(r["batch"]),
        "logdens": math.log10(r["dens"]),
        "logsyn": math.log10(r["syn"]),
    }


def featurize_b(r):
    return {
        "logn": math.log2(r["n"]),
        "logbatch": math.log2(r["batch"]),
        "logdens": math.log10(r["dens"]),
    }


def gini(labels):
    if not labels:
        return 0.0
    p = sum(labels) / len(labels)
    return 1 - p * p - (1 - p) * (1 - p)


def best_split(data, labels, features, min_leaf):
    n = len(data)
    base = gini(labels)
    best = None
    for feat in features:
        vals = sorted({d[feat] for d in data})
        for i in range(len(vals) - 1):
            thresh = (vals[i] + vals[i + 1]) / 2
            left_labels = [lb for d, lb in zip(data, labels, strict=False) if d[feat] <= thresh]
            right_labels = [lb for d, lb in zip(data, labels, strict=False) if d[feat] > thresh]
            if len(left_labels) < min_leaf or len(right_labels) < min_leaf:
                continue
            g = (len(left_labels) * gini(left_labels) + len(right_labels) * gini(right_labels)) / n
            gain = base - g
            if best is None or gain > best[0] + 1e-12:
                best = (gain, feat, thresh)
    return best


def fit(data, labels, features, max_depth, min_leaf, min_gain):
    node = Node()
    node.n = len(data)
    p = sum(labels) / len(labels) if labels else 0
    node.pred = p >= 0.5
    node.acc = max(p, 1 - p) if labels else 1.0
    if max_depth == 0 or len(set(labels)) <= 1 or len(data) < 2 * min_leaf:
        return node
    split = best_split(data, labels, features, min_leaf)
    if split is None or split[0] < min_gain:
        return node
    _gain, feat, thresh = split
    node.feat, node.thresh = feat, thresh
    li = [i for i, d in enumerate(data) if d[feat] <= thresh]
    ri = [i for i, d in enumerate(data) if d[feat] > thresh]
    node.left = fit([data[i] for i in li], [labels[i] for i in li], features, max_depth - 1, min_leaf, min_gain)
    node.right = fit([data[i] for i in ri], [labels[i] for i in ri], features, max_depth - 1, min_leaf, min_gain)
    return node


def predict(node, d):
    if node.feat is None:
        return node.pred
    if d[node.feat] <= node.thresh:
        return predict(node.left, d)
    return predict(node.right, d)


def accuracy(node, data, labels):
    return sum(1 for d, lb in zip(data, labels, strict=False) if predict(node, d) == lb) / len(data)


def count_leaves(node):
    if node.feat is None:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


def prune_redundant(node):
    """Collapse any split whose two children (after recursively pruning
    them) are both leaves predicting the same boolean -- the split changes
    nothing about the decision, so merging it is lossless. Repeated
    bottom-up to fixpoint via the recursion itself."""
    if node.feat is None:
        return node
    node.left = prune_redundant(node.left)
    node.right = prune_redundant(node.right)
    if node.left.feat is None and node.right.feat is None and node.left.pred == node.right.pred:
        merged = Node()
        merged.pred = node.left.pred
        merged.n = node.left.n + node.right.n
        merged.acc = (node.left.acc * node.left.n + node.right.acc * node.right.n) / merged.n if merged.n else 1.0
        return merged
    return node


FNAME = {"logn": "n", "logbatch": "batch", "logdens": "density", "logsyn": "syn_density"}


def print_tree(node, depth=0):
    pad = "  " * depth
    if node.feat is None:
        print(f"{pad}-> {node.pred}  (n={node.n}, leaf_acc={node.acc:.3f})")
        return
    fname = FNAME[node.feat]
    thresh_real = 2**node.thresh if node.feat in ("logn", "logbatch") else 10**node.thresh
    print(f"{pad}if {fname} <= {thresh_real:.6g} (n={node.n}):")
    print_tree(node.left, depth + 1)
    print(f"{pad}else:  # {fname} > {thresh_real:.6g}")
    print_tree(node.right, depth + 1)


def to_cpp(node, cvar_map, depth=1):
    """cvar_map: feat -> C++ expression string (e.g. 'batch', 'density', 'n', 'syn_density')."""
    pad = "    " * depth
    if node.feat is None:
        return f"{pad}return {'true' if node.pred else 'false'};\n"
    cexpr = cvar_map[node.feat]
    thresh_str = f"{2**node.thresh:.4f}" if node.feat in ("logn", "logbatch") else f"{10**node.thresh:.6f}"
    out = f"{pad}if ({cexpr} <= {thresh_str}f) {{\n"
    out += to_cpp(node.left, cvar_map, depth + 1)
    out += f"{pad}}} else {{\n"
    out += to_cpp(node.right, cvar_map, depth + 1)
    out += f"{pad}}}\n"
    return out


def kfold_indices(n, k, seed=0):
    idx = list(range(n))
    # fold assignment, not security-sensitive
    random.Random(seed).shuffle(idx)  # noqa: S311  # nosec B311
    return [idx[i::k] for i in range(k)]


def cv_accuracy(data, labels, features, max_depth, min_leaf, min_gain, k=5, seed=0):
    n = len(data)
    folds = kfold_indices(n, k, seed)
    accs = []
    for i in range(k):
        test_idx = set(folds[i])
        train_data = [data[j] for j in range(n) if j not in test_idx]
        train_labels = [labels[j] for j in range(n) if j not in test_idx]
        test_data = [data[j] for j in range(n) if j in test_idx]
        test_labels = [labels[j] for j in range(n) if j in test_idx]
        if len(set(train_labels)) <= 1 or not test_data:
            continue
        tree = fit(train_data, train_labels, features, max_depth, min_leaf, min_gain)
        accs.append(accuracy(tree, test_data, test_labels))
    return sum(accs) / len(accs)


# (max_depth, min_leaf, min_gain) grid. min_leaf floor of 6 deliberately
# excludes the noise-fitting region found at min_leaf=3 (see module docstring).
GRID = [(d, ml, mg) for d in (3, 4, 5, 6, 8) for ml in (6, 10, 16, 24) for mg in (0.005, 0.01, 0.02)]


def run(name, rows, ops, target_label, featurize_fn, features, cvar_map, k=5, verbose=False):
    sub = [r for r in rows if r["op"] in ops]
    data = [featurize_fn(r) for r in sub]
    labels = [r["best"] == target_label for r in sub]
    baseline = max(sum(labels), len(labels) - sum(labels)) / len(labels)

    best = None
    for max_depth, min_leaf, min_gain in GRID:
        cv_acc = cv_accuracy(data, labels, features, max_depth, min_leaf, min_gain, k)
        if verbose:
            print(f"  depth={max_depth} min_leaf={min_leaf} min_gain={min_gain}: CV_acc={cv_acc:.4f}")
        if best is None or cv_acc > best[0]:
            best = (cv_acc, max_depth, min_leaf, min_gain)
    cv_acc, max_depth, min_leaf, min_gain = best

    tree = fit(data, labels, features, max_depth, min_leaf, min_gain)
    tree = prune_redundant(tree)
    train_acc = accuracy(tree, data, labels)
    leaves = count_leaves(tree)

    print(f"\n########## {name} ##########")
    print(
        f"n={len(sub)}  baseline={baseline:.4f}  cv_acc={cv_acc:.4f}  train_acc={train_acc:.4f}  "
        f"leaves={leaves}  (depth={max_depth}, min_leaf={min_leaf}, min_gain={min_gain})"
    )
    print_tree(tree)
    print("\n--- C++ ---")
    print(to_cpp(tree, cvar_map))
    return tree, cv_acc, train_acc, leaves


# ── Oblique (linear-combination) splits -- opt-in comparison, NOT used to
# generate engine_select.hpp. Tested 2026-09-17 (motivated by the fitted
# trees' crossover boundaries looking "a little bit diagonal" in a plotted
# report): at EVERY node an oblique split is a strict superset of the axis-
# aligned candidates (weight 1.0 or 0.0 degenerates to a plain axis split),
# so greedy training-set fit can only improve, never worsen -- confirmed,
# training accuracy rose on every rule when oblique was enabled. But CV
# (held-out) accuracy is a different question: a richer per-node candidate
# search is a higher-variance fitting PROCEDURE at any fixed (max_depth,
# min_leaf, min_gain), since those don't bound the extra implicit search
# (~60+ candidate directions/node here vs ~4 for axis-only). Result, same
# CV grid search run independently for each (so hyperparameters aren't
# unfairly reused across model classes), heavier min_leaf up to 64 included
# to check whether more regularization would close the gap (it didn't --
# selected min_leaf never left the original 6-24 range): Group A forward
# 92.6%->90.3%, Group A backward 91.6%->91.3%, Group B forward 93.9%-
# >94.3%, Group B backward 86.5%->84.5%. Net: one marginal, plausibly-
# noise gain, three real losses -- exactly the known small-sample
# generalization cost of oblique/multivariate CART (Breiman's original
# work already noted this; it's why oblique trees never displaced
# axis-aligned CART despite being strictly more expressive). Kept here as
# a working alternative for a future re-fit with much more data, where
# this tradeoff could plausibly flip -- run with --compare-oblique.
OBLIQUE_WEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def standardize(data, features):
    means = {f: sum(d[f] for d in data) / len(data) for f in features}
    stds = {}
    for f in features:
        var = sum((d[f] - means[f]) ** 2 for d in data) / len(data)
        stds[f] = var**0.5 if var > 1e-12 else 1.0
    return [{f: (d[f] - means[f]) / stds[f] for f in features} for d in data]


def _project(d, feat):
    if isinstance(feat, tuple):
        _, fi, fj, w = feat
        return w * d[fi] + (1 - w) * d[fj]
    return d[feat]


def best_split_oblique(data, labels, features, min_leaf):
    """Like best_split, plus every unordered feature pair at a grid of
    mixing weights, projected onto that direction and thresholded the
    same way. Oblique candidates are keyed ('OBLIQUE', fi, fj, w)."""
    n = len(data)
    base = gini(labels)
    best = best_split(data, labels, features, min_leaf)

    for fi_idx, fi in enumerate(features):
        for fj in features[fi_idx + 1 :]:
            for w in OBLIQUE_WEIGHTS:
                feat = ("OBLIQUE", fi, fj, w)
                proj = [_project(d, feat) for d in data]
                vals = sorted(set(proj))
                for i in range(len(vals) - 1):
                    thresh = (vals[i] + vals[i + 1]) / 2
                    left = [lb for p, lb in zip(proj, labels, strict=False) if p <= thresh]
                    right = [lb for p, lb in zip(proj, labels, strict=False) if p > thresh]
                    if len(left) < min_leaf or len(right) < min_leaf:
                        continue
                    g = (len(left) * gini(left) + len(right) * gini(right)) / n
                    gain = base - g
                    if best is None or gain > best[0] + 1e-12:
                        best = (gain, feat, thresh)
    return best


def fit_oblique(data, labels, features, max_depth, min_leaf, min_gain):
    node = Node()
    node.n = len(data)
    p = sum(labels) / len(labels) if labels else 0
    node.pred = p >= 0.5
    node.acc = max(p, 1 - p) if labels else 1.0
    if max_depth == 0 or len(set(labels)) <= 1 or len(data) < 2 * min_leaf:
        return node
    split = best_split_oblique(data, labels, features, min_leaf)
    if split is None or split[0] < min_gain:
        return node
    _gain, feat, thresh = split
    node.feat, node.thresh = feat, thresh
    li = [i for i, d in enumerate(data) if _project(d, feat) <= thresh]
    ri = [i for i, d in enumerate(data) if _project(d, feat) > thresh]
    node.left = fit_oblique([data[i] for i in li], [labels[i] for i in li], features, max_depth - 1, min_leaf, min_gain)
    node.right = fit_oblique(
        [data[i] for i in ri], [labels[i] for i in ri], features, max_depth - 1, min_leaf, min_gain
    )
    return node


def predict_oblique(node, d):
    if node.feat is None:
        return node.pred
    if _project(d, node.feat) <= node.thresh:
        return predict_oblique(node.left, d)
    return predict_oblique(node.right, d)


def accuracy_oblique(node, data, labels):
    return sum(1 for d, lb in zip(data, labels, strict=False) if predict_oblique(node, d) == lb) / len(data)


def cv_accuracy_oblique(data, labels, features, max_depth, min_leaf, min_gain, k=5, seed=0):
    n = len(data)
    folds = kfold_indices(n, k, seed)
    accs = []
    for i in range(k):
        test_idx = set(folds[i])
        train_data = [data[j] for j in range(n) if j not in test_idx]
        train_labels = [labels[j] for j in range(n) if j not in test_idx]
        test_data = [data[j] for j in range(n) if j in test_idx]
        test_labels = [labels[j] for j in range(n) if j in test_idx]
        if len(set(train_labels)) <= 1 or not test_data:
            continue
        tree = fit_oblique(train_data, train_labels, features, max_depth, min_leaf, min_gain)
        accs.append(accuracy_oblique(tree, test_data, test_labels))
    return sum(accs) / len(accs)


OBLIQUE_GRID = [
    (d, ml, mg) for d in (2, 3, 4, 5, 6, 8) for ml in (6, 10, 16, 24, 32, 48, 64) for mg in (0.005, 0.01, 0.02)
]


def compare_oblique(name, rows, ops, target_label, featurize_fn, features):
    sub = [r for r in rows if r["op"] in ops]
    raw_data = [featurize_fn(r) for r in sub]
    labels = [r["best"] == target_label for r in sub]
    data = standardize(raw_data, features)

    axis_best = max(
        ((cv_accuracy(data, labels, features, d, ml, mg), d, ml, mg) for d, ml, mg in OBLIQUE_GRID),
        key=lambda t: t[0],
    )
    oblique_best = max(
        ((cv_accuracy_oblique(data, labels, features, d, ml, mg), d, ml, mg) for d, ml, mg in OBLIQUE_GRID),
        key=lambda t: t[0],
    )
    print(f"\n########## {name} (oblique comparison) ##########")
    print(f"  axis-only best:    CV acc={axis_best[0]:.4f}  {axis_best[1:]}")
    print(f"  axis+oblique best: CV acc={oblique_best[0]:.4f}  {oblique_best[1:]}")
    print(f"  delta={oblique_best[0] - axis_best[0]:+.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare-oblique",
        action="store_true",
        help="also run the oblique-split CV comparison (slow, ~4 min) -- see module docstring for the "
        "2026-09-17 negative result; does not affect the generated engine_select.hpp rules",
    )
    args = parser.parse_args()

    group_a, group_b = load_rows()
    a_features = ["logn", "logbatch", "logdens", "logsyn"]
    b_features = ["logn", "logbatch", "logdens"]
    a_cvar = {"logn": "n", "logbatch": "batch", "logdens": "density", "logsyn": "syn_density"}
    b_cvar = {"logn": "n", "logbatch": "batch", "logdens": "density"}

    run("Group A FORWARD (sisldo)", group_a, {"fwd"}, "sisldo", featurize_a, a_features, a_cvar)
    run("Group A BACKWARD (sisldo)", group_a, {"bwd0", "bwdX"}, "sisldo", featurize_a, a_features, a_cvar)
    run("Group B FORWARD (sidldo)", group_b, {"fwd"}, "sidldo", featurize_b, b_features, b_cvar)
    run("Group B BACKWARD (sidldo)", group_b, {"bwd0", "bwdX"}, "sidldo", featurize_b, b_features, b_cvar)

    if args.compare_oblique:
        compare_oblique("Group A FORWARD", group_a, {"fwd"}, "sisldo", featurize_a, a_features)
        compare_oblique("Group A BACKWARD", group_a, {"bwd0", "bwdX"}, "sisldo", featurize_a, a_features)
        compare_oblique("Group B FORWARD", group_b, {"fwd"}, "sidldo", featurize_b, b_features)
        compare_oblique("Group B BACKWARD", group_b, {"bwd0", "bwdX"}, "sidldo", featurize_b, b_features)
