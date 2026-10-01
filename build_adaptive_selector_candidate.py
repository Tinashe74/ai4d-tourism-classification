from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from ordinal_attack import champion_oof
from reproducibility_recovery import fit_vec, apply, norm, data, cross

ROOT = Path(__file__).resolve().parent
CLASSES = ["High Cost", "Higher Cost", "Highest Cost", "Low Cost", "Lower Cost", "Normal Cost"]
SEEDS = [2025, 2026, 42, 777, 3407]
FLOOR = 0.7


def trip_components():
    _, br, hy = data()
    champ = champion_oof()
    h = np.load(ROOT / "hierarchical_candidate_oof.npy")
    l = np.load(ROOT / "local_similarity_oof.npy")
    z = np.log(np.clip(.89 * champ + .02 * h + .09 * l, 1e-15, 1)) / .98
    z -= z.max(1, keepdims=True)
    base = norm(np.exp(z))
    struct = np.mean([np.load(ROOT / f) for f in [
        "xgb_compact_struct_oof_seed2026.npy",
        "xgb_compact_struct_oof_seed42.npy",
        "xgb_compact_struct_oof_seed3407.npy",
    ]], axis=0)
    trip = norm(.85 * base + .15 * struct)
    return br, trip, hy


def prediction_features(P):
    features = []
    for p in P:
        order = np.sort(p, axis=1)[:, ::-1]
        entropy = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)[:, None]
        features.extend([p, entropy, order[:, 0:1] - order[:, 1:2], order[:, 0:1]])
    for j in range(1, len(P)):
        features.append(P[j] - P[0])
    return np.hstack(features)


def gate_oof(P, y, seed):
    X = prediction_features(P)
    out = np.zeros((len(y), 6))
    weights = []
    fold_scores = []
    losses = np.stack([
        -np.log(np.clip(P[j][np.arange(len(y)), y], 1e-12, 1))
        for j in range(len(P))
    ], axis=1)
    winner = losses.argmin(1)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        model = LogisticRegression(C=.1, max_iter=120, solver="lbfgs", random_state=seed)
        model.fit(X[tr], winner[tr])
        gw = model.predict_proba(X[va])
        full = np.zeros((len(va), len(P)))
        full[:, 0] = FLOOR
        full += (1 - FLOOR) * gw
        q = np.einsum("nm,mnc->nc", full, np.stack([P[j][va] for j in range(len(P))]))
        out[va] = norm(q)
        weights.append(full)
        fold_scores.append(log_loss(y[va], out[va], labels=np.arange(6)))
    return out, np.vstack(weights), fold_scores, winner


def fit_test_gate(P, y):
    X = prediction_features(P)
    losses = np.stack([
        -np.log(np.clip(P[j][np.arange(len(y)), y], 1e-12, 1))
        for j in range(len(P))
    ], axis=1)
    winner = losses.argmin(1)
    model = LogisticRegression(C=.1, max_iter=120, solver="lbfgs", random_state=2026)
    model.fit(X, winner)
    return model


def stats(weights):
    return {
        "mean": weights.mean(0),
        "median": np.median(weights, axis=0),
        "p05": np.quantile(weights, .05, axis=0),
        "p95": np.quantile(weights, .95, axis=0),
        "min": weights.min(0),
        "max": weights.max(0),
    }


def print_gate_stats(label, s):
    print(label)
    for k, v in s.items():
        print(k, np.round(v, 9).tolist())


def main():
    train = pd.read_csv(ROOT / "Train.csv")
    test = pd.read_csv(ROOT / "Test.csv")
    sample = pd.read_csv(ROOT / "SampleSubmission.csv")
    y = pd.factorize(train.cost_category, sort=True)[0]

    br, trip, hy = trip_components()
    cb, _ = cross(br, y, 2026)
    ch, _ = cross(hy, y, 2026)
    bcal = norm(.85 * cb + .15 * ch)
    P = np.stack([br, bcal, trip, hy])
    base_ll = log_loss(y, br, labels=np.arange(6))

    selector, oof_weights, fold_scores, winner = gate_oof(P, y, 2026)
    selector_ll = log_loss(y, selector, labels=np.arange(6))
    print("BASELINE", repr(base_ll))
    print("SELECTOR", repr(selector_ll))
    print("IMPROVEMENT", repr(base_ll - selector_ll))
    print("FOLD_SCORES", [repr(x) for x in fold_scores])

    seed_imps = []
    for seed in SEEDS:
        q, _, _, _ = gate_oof(P, y, seed)
        imp = base_ll - log_loss(y, q, labels=np.arange(6))
        seed_imps.append(imp)
        print("SEED", seed, "IMPROVEMENT", repr(imp), "OOF", repr(base_ll - imp))
    seed_imps = np.asarray(seed_imps)
    print("SEED_SUMMARY", {
        "mean": float(seed_imps.mean()),
        "std": float(seed_imps.std()),
        "min": float(seed_imps.min()),
        "max": float(seed_imps.max()),
    })

    print_gate_stats("OOF_GATE", stats(oof_weights))

    # OOF diagnostics: fold losses, per-class log loss, and hardest-base-relative slices.
    oof_class = []
    for k, name in enumerate(CLASSES):
        mask = y == k
        oof_class.append([
            name, int(mask.sum()),
            float(log_loss(y[mask], br[mask], labels=np.arange(6))),
            float(log_loss(y[mask], selector[mask], labels=np.arange(6))),
        ])
    print("PER_CLASS", oof_class)
    base_nll = -np.log(np.clip(br[np.arange(len(y)), y], 1e-12, 1))
    sel_nll = -np.log(np.clip(selector[np.arange(len(y)), y], 1e-12, 1))
    delta = base_nll - sel_nll
    print("WORST_SLICE_IMPROVEMENTS", {
        str(frac): float(np.mean(np.sort(delta)[-max(1, int(len(y) * frac)):]))
        for frac in [.005, .01, .05, .10, .20]
    })

    # Reconstruct the matching test-side components without changing the protected file.
    protected = pd.read_csv(ROOT / "candidate_xgboost_breakthrough_v3.csv")
    protected_hash_before = hashlib.sha256((ROOT / "candidate_xgboost_breakthrough_v3.csv").read_bytes()).hexdigest()
    brt = protected[CLASSES].to_numpy(float)
    raw_t = np.mean([np.load(ROOT / f) for f in [
        "xgb_compact_test.npy", "xgb_compact_test_seed42.npy", "xgb_compact_test_seed3407.npy"
    ]], axis=0)
    struct_t = np.mean([np.load(ROOT / f) for f in [
        "xgb_compact_struct_test_seed2026.npy", "xgb_compact_struct_test_seed42.npy",
        "xgb_compact_struct_test_seed3407.npy"
    ]], axis=0)
    hyb_t = np.mean([np.load(ROOT / f) for f in [
        "xgb_compact_hybrid_test_seed2026.npy", "xgb_compact_hybrid_test_seed42.npy",
        "xgb_compact_hybrid_test_seed777.npy", "xgb_compact_hybrid_test_seed3407.npy"
    ]], axis=0)
    h_t = np.load(ROOT / "hierarchical_model_test_seed2026.npy")
    l_t = np.load(ROOT / "local_similarity_test.npy")
    z = np.log(np.clip(.89 * brt + .02 * h_t + .09 * l_t, 1e-15, 1)) / .98
    z -= z.max(1, keepdims=True)
    bt = norm(np.exp(z))
    raw_t = norm(raw_t)
    struct_t = norm(struct_t)
    hyb_t = norm(hyb_t)
    bcal_t = norm(.85 * apply(brt, fit_vec(np.log(np.clip(br, 1e-12, 1)), y))
                   + .15 * apply(hyb_t, fit_vec(np.log(np.clip(hy, 1e-12, 1)), y)))
    PT = np.stack([brt, bcal_t, norm(.85 * bt + .15 * struct_t), hyb_t])
    test_model = fit_test_gate(P, y)
    test_gate = np.zeros((len(test), len(P)))
    test_gate[:, 0] = FLOOR
    test_gate += (1 - FLOOR) * test_model.predict_proba(prediction_features(PT))
    test_pred = norm(np.einsum("nm,mnc->nc", test_gate, PT))
    print_gate_stats("TEST_GATE", stats(test_gate))
    oof_s = stats(oof_weights)
    test_s = stats(test_gate)
    mean_diff = np.abs(test_s["mean"] - oof_s["mean"])
    p05_diff = np.abs(test_s["p05"] - oof_s["p05"])
    p95_diff = np.abs(test_s["p95"] - oof_s["p95"])
    mismatch = bool(np.max(mean_diff) > .05 or np.max(p05_diff) > .10 or np.max(p95_diff) > .10)
    print("GATE_MEAN_ABS_DIFF", np.round(mean_diff, 9).tolist())
    print("GATE_MISMATCH", mismatch)
    if mismatch:
        print("TEST GATE DISTRIBUTION MISMATCH")
        return

    # Use the official sample submission as the sole schema, ID source, and row-order source.
    schema_ok = list(sample.columns) == [sample.columns[0]] + CLASSES
    ids_ok = len(sample) == 6169 and sample.iloc[:, 0].is_unique and sample.iloc[:, 0].equals(test.iloc[:, 0])
    finite_ok = bool(np.isfinite(test_pred).all())
    range_ok = bool((test_pred >= 0).all() and (test_pred <= 1).all())
    sums_ok = bool(np.allclose(test_pred.sum(1), 1.0, rtol=0, atol=1e-12))
    if not (schema_ok and ids_ok and finite_ok and range_ok and sums_ok):
        print("PROBABILITY_INTEGRITY_FAILED", {
            "schema": schema_ok, "ids": ids_ok, "finite": finite_ok,
            "range": range_ok, "sums": sums_ok,
        })
        return

    out = sample.copy()
    out[CLASSES] = test_pred
    output = ROOT / "candidate_adaptive_selector_v1.csv"
    out.to_csv(output, index=False)
    protected_hash_after = hashlib.sha256((ROOT / "candidate_xgboost_breakthrough_v3.csv").read_bytes()).hexdigest()
    consensus_hash = hashlib.sha256((ROOT / "candidate_consensus_correction_final.csv").read_bytes()).hexdigest()
    written = pd.read_csv(output)
    written_probs = written[CLASSES].to_numpy(float)
    print("FINAL", {
        "file": output.name,
        "rows": len(written),
        "schema": list(written.columns) == list(sample.columns),
        "ids": written.iloc[:, 0].equals(sample.iloc[:, 0]),
        "duplicates": int(written.iloc[:, 0].duplicated().sum()),
        "missing": int(written.isna().sum().sum()),
        "finite": bool(np.isfinite(written_probs).all()),
        "range": bool((written_probs >= 0).all() and (written_probs <= 1).all()),
        "row_sum_max_error": float(np.max(np.abs(written_probs.sum(1) - 1))),
        "protected_unchanged": protected_hash_before == protected_hash_after,
        "protected_hash": protected_hash_after,
        "consensus_hash": consensus_hash,
    })


if __name__ == "__main__":
    main()
