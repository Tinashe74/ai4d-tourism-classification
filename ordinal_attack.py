from pathlib import Path
import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from experiment import features

ROOT = Path(__file__).resolve().parent
SEED = int(os.environ.get("ORDINAL_SEED", "2026"))
ECONOMIC = ["Low Cost", "Lower Cost", "Normal Cost", "High Cost", "Higher Cost", "Highest Cost"]


def champion_oof():
    files = ["geo_frequency", "onehot", "interaction", "catboost", "joint_frequency", "joint_frequency_seed2024", "joint_frequency_seed3407"]
    arrays = np.stack([np.load(ROOT / (f + "_oof.npy")) for f in files])
    weights = np.array([.0413259998, .2217763817, .1500358034, .1044899380, .0678346740, .1734401381, .2410970651])
    p = np.einsum("m,mnc->nc", weights, arrays)
    z = np.log(np.clip(p, 1e-15, 1)) / .9693158908; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    correction = np.mean([np.load(ROOT / f) for f in ["consensus_correction_oof.npy", "consensus_correction_alt_oof.npy", "consensus_correction_seed42_oof.npy", "consensus_correction_seed777_oof.npy"]], axis=0)
    return .7 * p + .3 * correction


def probabilities(cumulative):
    out = np.zeros((len(cumulative[0]), 6))
    out[:, 0] = 1 - cumulative[0]
    for index in range(1, 5):
        out[:, index] = cumulative[index - 1] - cumulative[index]
    out[:, 5] = cumulative[4]
    return np.clip(out, 1e-12, None) / np.clip(out, 1e-12, None).sum(axis=1, keepdims=True)


def main():
    train, test = pd.read_csv(ROOT / "Train.csv"), pd.read_csv(ROOT / "Test.csv")
    labels = train.cost_category.to_numpy()
    economic_y = np.array([ECONOMIC.index(value) for value in labels])
    joined = pd.concat([features(train), features(test)], ignore_index=True)
    cats = joined.select_dtypes(include=["object", "string"]).columns.tolist()
    joined = pd.get_dummies(joined, columns=cats, dummy_na=True).fillna(-1); joined.columns = [f"f{i}" for i in range(joined.shape[1])]
    x, xt = joined.iloc[:len(train)], joined.iloc[len(train):]
    oof_cum = np.zeros((5, len(train))); test_cum = np.zeros((5, len(test)))
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for threshold in range(5):
        binary = (economic_y > threshold).astype(int)
        for fold, (a, b) in enumerate(folds.split(x, economic_y), 1):
            model = LGBMClassifier(objective="binary", n_estimators=900, num_leaves=25,
                                   min_child_samples=60, learning_rate=.025, reg_lambda=10,
                                   colsample_bytree=.85, subsample=.9, subsample_freq=1,
                                   random_state=SEED + threshold * 10 + fold, n_jobs=4, verbosity=-1)
            model.fit(x.iloc[a], binary[a], eval_set=[(x.iloc[b], binary[b])], callbacks=[early_stopping(70, verbose=False)])
            oof_cum[threshold, b] = model.predict_proba(x.iloc[b])[:, 1]
            test_cum[threshold] += model.predict_proba(xt)[:, 1] / 5
        print("threshold", threshold, "positive", binary.mean())
    ordinal_oof = probabilities(oof_cum); ordinal_test = probabilities(test_cum)
    target = pd.factorize(train.cost_category, sort=True)[0]
    order = [ECONOMIC.index(x) for x in sorted(ECONOMIC)]
    # Reorder economic probabilities to the competition's alphabetical class order.
    alphabetical = sorted(ECONOMIC)
    ordinal_oof_comp = ordinal_oof[:, [ECONOMIC.index(x) for x in alphabetical]]
    ordinal_test_comp = ordinal_test[:, [ECONOMIC.index(x) for x in alphabetical]]
    print("ordinal score", log_loss(target, ordinal_oof_comp, labels=np.arange(6)))
    base = champion_oof(); print("champion", log_loss(target, base, labels=np.arange(6)))
    for alpha in [.01, .025, .05, .075, .1, .15, .2, .25]:
        print("alpha", alpha, log_loss(target, (1-alpha)*base + alpha*ordinal_oof_comp, labels=np.arange(6)))
    np.save(ROOT / f"ordinal_attack_seed{SEED}_oof.npy", ordinal_oof_comp); np.save(ROOT / f"ordinal_attack_seed{SEED}_test.npy", ordinal_test_comp)


if __name__ == "__main__": main()
