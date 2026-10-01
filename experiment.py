from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent
TARGET = "cost_category"
ID = "Tour_ID"
SEED = 42


def features(frame):
    x = frame.drop(columns=[TARGET, ID], errors="ignore").copy()
    binary = [c for c in x if c.startswith("package_")]
    for c in binary:
        x[c + "_num"] = x[c].map({"Yes": 1, "No": 0}).fillna(-1)
    x["total_nights"] = x.night_mainland + x.night_zanzibar
    x["party_size"] = x.total_female.fillna(0) + x.total_male.fillna(0)
    x["package_count"] = x[[c + "_num" for c in binary]].clip(lower=0).sum(axis=1)
    x["zanzibar_share"] = x.night_zanzibar / (x.total_nights + 1)
    x["female_share"] = x.total_female.fillna(0) / (x["party_size"] + 1)
    x["night_ratio"] = x.night_zanzibar / (x.night_mainland + 1)
    x["country_purpose"] = x.country.astype(str) + "__" + x.purpose.astype(str)
    x["country_activity"] = x.country.astype(str) + "__" + x.main_activity.astype(str)
    return x


def main():
    train = pd.read_csv(ROOT / "Train.csv")
    test = pd.read_csv(ROOT / "Test.csv")
    x, xt = features(train), features(test)
    y, names = pd.factorize(train[TARGET], sort=True)
    cats = x.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in cats:
        all_values = pd.concat([x[c], xt[c]]).fillna("__MISSING__").astype(str)
        codes, _ = pd.factorize(all_values)
        x[c] = codes[:len(x)]
        xt[c] = codes[len(x):]
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    categorical_indices = [x.columns.get_loc(c) for c in cats]
    configs = [
        dict(num_leaves=31, learning_rate=.025, min_child_samples=35, subsample=.85, colsample_bytree=.85, reg_lambda=3),
        dict(num_leaves=20, learning_rate=.03, min_child_samples=25, subsample=.9, colsample_bytree=.9, reg_lambda=5),
        dict(num_leaves=45, learning_rate=.02, min_child_samples=45, subsample=.85, colsample_bytree=.8, reg_lambda=5),
    ]
    predictions, test_predictions = [], []
    for ci, cfg in enumerate(configs):
        oof = np.zeros((len(x), len(names)))
        pred = np.zeros((len(xt), len(names)))
        for fold, (a, b) in enumerate(folds.split(x, y), 1):
            model = LGBMClassifier(objective="multiclass", num_class=len(names), n_estimators=1600,
                                    random_state=SEED + fold, n_jobs=4, verbosity=-1, **cfg)
            model.fit(x.iloc[a], y[a], categorical_feature=categorical_indices,
                      eval_set=[(x.iloc[b], y[b])], callbacks=[early_stopping(100, verbose=False)])
            oof[b] = model.predict_proba(x.iloc[b])
            pred += model.predict_proba(xt) / folds.n_splits
        score = log_loss(y, oof, labels=np.arange(len(names)))
        print("lgb", ci, score)
        predictions.append(oof); test_predictions.append(pred)
    scores = [log_loss(y, p, labels=np.arange(len(names))) for p in predictions]
    weights = np.exp(-20 * (np.array(scores) - min(scores)))
    weights /= weights.sum()
    print("scores", scores, "weights", weights.tolist())
    final = sum(weight * pred for weight, pred in zip(weights, test_predictions))
    out = pd.DataFrame(final, columns=names); out.insert(0, ID, test[ID].values); out.to_csv(ROOT / "candidate_submission.csv", index=False)


if __name__ == "__main__": main()
