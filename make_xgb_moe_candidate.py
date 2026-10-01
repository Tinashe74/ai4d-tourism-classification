from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from xgb_compact_oof import prepare

ROOT = Path(__file__).resolve().parent
CLASSES = ["High Cost", "Higher Cost", "Highest Cost", "Low Cost", "Lower Cost", "Normal Cost"]
OUT = ROOT / "candidate_xgb_moe_southafrica_business_v1.csv"
CHAMPION = ROOT / "candidate_xgboost_breakthrough_v3.csv"
CONSENSUS = ROOT / "candidate_consensus_correction_final.csv"


def norm(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def main():
    train = pd.read_csv(ROOT / "Train.csv")
    test = pd.read_csv(ROOT / "Test.csv")
    sample = pd.read_csv(ROOT / "SampleSubmission.csv")
    champion = pd.read_csv(CHAMPION)
    if list(sample.columns) != [sample.columns[0]] + CLASSES or list(champion.columns) != list(sample.columns):
        raise ValueError("Schema mismatch")
    if not sample.iloc[:, 0].equals(test.iloc[:, 0]) or not sample.iloc[:, 0].equals(champion.iloc[:, 0]):
        raise ValueError("ID order mismatch")
    x, xt = prepare(train, test)
    champion_test = champion[CLASSES].to_numpy(float)
    nights_tr = train.night_mainland.fillna(0) + train.night_zanzibar.fillna(0)
    nights_te = test.night_mainland.fillna(0) + test.night_zanzibar.fillna(0)
    pkg_tr = train[[c for c in train.columns if c.startswith("package_")]].eq("Yes").sum(axis=1)
    pkg_te = test[[c for c in test.columns if c.startswith("package_")]].eq("Yes").sum(axis=1)
    regimes_tr = {
        "south_africa_package": (train.country == "SOUTH AFRICA") & (train.tour_arrangement == "Package Tour"),
        "business_conference": train.purpose.isin(["Business", "Meetings and Conference"]),
    }
    regimes_te = {
        "south_africa_package": (test.country == "SOUTH AFRICA") & (test.tour_arrangement == "Package Tour"),
        "business_conference": test.purpose.isin(["Business", "Meetings and Conference"]),
    }
    y = pd.factorize(train.cost_category, sort=True)[0]
    specialist_test = {}
    for name in regimes_tr:
        rows = np.flatnonzero(regimes_tr[name].to_numpy())
        test_rows = np.flatnonzero(regimes_te[name].to_numpy())
        model = XGBClassifier(
            n_estimators=260, max_depth=3, learning_rate=.055, min_child_weight=8,
            subsample=.9, colsample_bytree=.9, reg_alpha=.2, reg_lambda=10,
            objective="multi:softprob", num_class=6, eval_metric="mlogloss",
            tree_method="hist", n_jobs=2, random_state=2026,
        )
        model.fit(x.iloc[rows], y[rows], eval_set=[(x.iloc[rows], y[rows])], verbose=False)
        pred = np.zeros((len(test), 6))
        if len(test_rows):
            pred[test_rows] = norm(model.predict_proba(xt.iloc[test_rows]))
        specialist_test[name] = pred
        print(name, "train_rows", len(rows), "test_rows", len(test_rows), flush=True)

    # Exact best OOF configuration: sequential 10% membership corrections.
    pred = champion_test.copy()
    for name in ["south_africa_package", "business_conference"]:
        mask = regimes_te[name].to_numpy()
        pred[mask] = .9 * pred[mask] + .1 * specialist_test[name][mask]
    pred = norm(pred)
    out = sample.copy(); out[CLASSES] = pred
    arr = out[CLASSES].to_numpy(float)
    before = (hashlib.sha256(CHAMPION.read_bytes()).hexdigest(), hashlib.sha256(CONSENSUS.read_bytes()).hexdigest())
    checks = {
        "rows": len(out) == 6169,
        "schema": list(out.columns) == list(sample.columns),
        "ids": out.iloc[:, 0].equals(sample.iloc[:, 0]),
        "duplicates": int(out.iloc[:, 0].duplicated().sum()),
        "missing": int(out.isna().sum().sum()),
        "finite": bool(np.isfinite(arr).all()),
        "range": bool(((arr >= 0) & (arr <= 1)).all()),
        "row_sum_max_error": float(np.max(np.abs(arr.sum(1) - 1))),
        "changed_rows": int(np.any(np.abs(pred - champion_test) > 1e-12, axis=1).sum()),
        "mean_abs_move": float(np.abs(pred - champion_test).mean()),
    }
    if not checks["rows"] or not checks["schema"] or not checks["ids"] or checks["duplicates"] or checks["missing"] or not checks["finite"] or not checks["range"] or checks["row_sum_max_error"] > 1e-12:
        raise ValueError(checks)
    out.to_csv(OUT, index=False)
    after = (hashlib.sha256(CHAMPION.read_bytes()).hexdigest(), hashlib.sha256(CONSENSUS.read_bytes()).hexdigest())
    checks["protected_unchanged"] = before == after
    checks["class_means"] = pred.mean(0).round(8).tolist()
    print({"file": OUT.name, **checks})


if __name__ == "__main__":
    main()
