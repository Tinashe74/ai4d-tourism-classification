from pathlib import Path
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from experiment import features
import os

ROOT = Path(__file__).resolve().parent
SEED = int(os.environ.get("XGB_SEED", "2026"))
STRUCT = os.environ.get("XGB_STRUCT", "0") == "1"
HYBRID = os.environ.get("XGB_HYBRID", "0") == "1"

def prepare(train, test):
    a = features(train).copy(); b = features(test).copy()
    both = pd.concat([a, b], ignore_index=True)
    if STRUCT:
        both["total_nights_struct"] = both["night_mainland"] + both["night_zanzibar"]
        both["party_size_struct"] = both["total_female"].fillna(0) + both["total_male"].fillna(0)
        pkg = [c for c in both.columns if c.endswith("_num") and c.startswith("package_")]
        both["package_count_struct"] = both[pkg].clip(lower=0).sum(axis=1)
        both["zanzibar_share_struct"] = both["night_zanzibar"] / (both["total_nights_struct"] + 1)
        both["nights_per_person_struct"] = both["total_nights_struct"] / (both["party_size_struct"] + 1)
        both["party_per_night_struct"] = both["party_size_struct"] / (both["total_nights_struct"] + 1)
        both["female_share_struct"] = both["total_female"].fillna(0) / (both["party_size_struct"] + 1)
    if HYBRID:
        total = both["night_mainland"] + both["night_zanzibar"]
        duration_band = pd.cut(total, bins=[-1, 0, 3, 7, 14, 30, 999], labels=False).fillna(-1).astype(str)
        for left, name in [("purpose", "purpose_duration"), ("package_accomodation", "accommodation_duration"), ("main_activity", "activity_duration")]:
            both[name] = both[left].fillna("__MISSING__").astype(str) + "__" + duration_band
        both["country_arrangement"] = both["country"].fillna("__MISSING__").astype(str) + "__" + both["tour_arrangement"].fillna("__MISSING__").astype(str)
        both["country_accommodation"] = both["country"].fillna("__MISSING__").astype(str) + "__" + both["package_accomodation"].fillna("__MISSING__").astype(str)
        pkg_cols = [c for c in both.columns if c.startswith("package_") and not c.endswith("_num")]
        both["package_signature"] = both[pkg_cols].fillna("__MISSING__").astype(str).agg("|".join, axis=1)
    cats = both.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cats:
        vals, _ = pd.factorize(both[col].fillna("__MISSING__").astype(str), sort=True)
        both[col] = vals.astype(np.int32)
    both = both.replace([np.inf, -np.inf], np.nan).fillna(-1).astype(np.float32)
    return both.iloc[:len(a)].reset_index(drop=True), both.iloc[len(a):].reset_index(drop=True)

def main():
    start = time.time()
    train = pd.read_csv(ROOT / "Train.csv"); test = pd.read_csv(ROOT / "Test.csv")
    y = pd.factorize(train.cost_category, sort=True)[0]
    x, xt = prepare(train, test)
    print("rows", len(x), "features", x.shape[1], "memory_mb", round((x.memory_usage(deep=True).sum()+xt.memory_usage(deep=True).sum())/1e6, 1))
    oof = np.zeros((len(x), 6)); pred = np.zeros((len(xt), 6)); fold_scores = []
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(folds.split(x, y), 1):
        model = XGBClassifier(
            n_estimators=700, max_depth=4, learning_rate=.04,
            min_child_weight=12, subsample=.82, colsample_bytree=.85,
            reg_alpha=.25, reg_lambda=12, objective="multi:softprob",
            num_class=6, eval_metric="mlogloss", tree_method="hist",
            n_jobs=6, random_state=SEED + fold, early_stopping_rounds=60,
        )
        model.fit(x.iloc[tr_idx], y[tr_idx], eval_set=[(x.iloc[va_idx], y[va_idx])], verbose=False)
        vp = model.predict_proba(x.iloc[va_idx]); tp = model.predict_proba(xt)
        vp = vp / vp.sum(1, keepdims=True); tp = tp / tp.sum(1, keepdims=True)
        oof[va_idx] = vp; pred += tp / 5
        score = log_loss(y[va_idx], vp, labels=np.arange(6)); fold_scores.append(score)
        print("fold", fold, "best_iteration", model.best_iteration, "logloss", score)
    print("standalone", log_loss(y, oof, labels=np.arange(6)), "folds", fold_scores, "runtime_sec", round(time.time()-start, 1))
    tag = "_hybrid" if HYBRID else ("_struct" if STRUCT else "")
    np.save(ROOT / f"xgb_compact{tag}_oof_seed{SEED}.npy", oof); np.save(ROOT / f"xgb_compact{tag}_test_seed{SEED}.npy", pred)

if __name__ == "__main__": main()
