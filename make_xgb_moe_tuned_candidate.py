from pathlib import Path
import hashlib
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgb_compact_oof import prepare

ROOT = Path(__file__).resolve().parent
CLASSES = ['High Cost', 'Higher Cost', 'Highest Cost', 'Low Cost', 'Lower Cost', 'Normal Cost']
OUT = ROOT / os.environ.get('MOE_OUT', 'candidate_xgb_moe_tuned_v1.csv')
CHAMPION = ROOT / 'candidate_xgboost_breakthrough_v3.csv'
CONSENSUS = ROOT / 'candidate_consensus_correction_final.csv'

def norm(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)

train = pd.read_csv(ROOT / 'Train.csv'); test = pd.read_csv(ROOT / 'Test.csv')
sample = pd.read_csv(ROOT / 'SampleSubmission.csv'); champion = pd.read_csv(CHAMPION)
if list(sample.columns) != [sample.columns[0]] + CLASSES or list(champion.columns) != list(sample.columns): raise ValueError('schema')
if not sample.iloc[:, 0].equals(test.iloc[:, 0]) or not sample.iloc[:, 0].equals(champion.iloc[:, 0]): raise ValueError('ids')
x, xt = prepare(train, test); y = pd.factorize(train.cost_category, sort=True)[0]
tr_masks = {'sa': ((train.country == 'SOUTH AFRICA') & (train.tour_arrangement == 'Package Tour')).to_numpy(), 'biz': train.purpose.isin(['Business', 'Meetings and Conference']).to_numpy()}
te_masks = {'sa': ((test.country == 'SOUTH AFRICA') & (test.tour_arrangement == 'Package Tour')).to_numpy(), 'biz': test.purpose.isin(['Business', 'Meetings and Conference']).to_numpy()}
special = {}
for name in ['sa', 'biz']:
    rows = np.flatnonzero(tr_masks[name]); test_rows = np.flatnonzero(te_masks[name])
    model = XGBClassifier(n_estimators=260, max_depth=3, learning_rate=.055, min_child_weight=8, subsample=.9, colsample_bytree=.9, reg_alpha=.2, reg_lambda=10, objective='multi:softprob', num_class=6, eval_metric='mlogloss', tree_method='hist', n_jobs=2, random_state=2026)
    model.fit(x.iloc[rows], y[rows], eval_set=[(x.iloc[rows], y[rows])], verbose=False)
    p = np.zeros((len(test), 6)); p[test_rows] = norm(model.predict_proba(xt.iloc[test_rows])); special[name] = p
    print(name, 'train_rows', len(rows), 'test_rows', len(test_rows), flush=True)
pred = champion[CLASSES].to_numpy(float)
for name, weight in [('sa', float(os.environ.get('MOE_W_SA', '.20'))), ('biz', float(os.environ.get('MOE_W_BIZ', '.15')))]:
    m = te_masks[name]; pred[m] = (1-weight) * pred[m] + weight * special[name][m]
pred = norm(pred)
out = sample.copy(); out[CLASSES] = pred; arr = out[CLASSES].to_numpy(float)
checks = {'rows': len(out) == 6169, 'schema': list(out.columns) == list(sample.columns), 'ids': out.iloc[:,0].equals(sample.iloc[:,0]), 'duplicates': int(out.iloc[:,0].duplicated().sum()), 'missing': int(out.isna().sum().sum()), 'finite': bool(np.isfinite(arr).all()), 'range': bool(((arr >= 0) & (arr <= 1)).all()), 'row_sum_max_error': float(np.max(np.abs(arr.sum(1)-1))), 'changed_rows': int(np.any(np.abs(pred-champion[CLASSES].to_numpy(float)) > 1e-12, axis=1).sum())}
if not all([checks['rows'], checks['schema'], checks['ids'], checks['duplicates'] == 0, checks['missing'] == 0, checks['finite'], checks['range'], checks['row_sum_max_error'] <= 1e-12]): raise ValueError(checks)
before = (hashlib.sha256(CHAMPION.read_bytes()).hexdigest(), hashlib.sha256(CONSENSUS.read_bytes()).hexdigest())
out.to_csv(OUT, index=False)
after = (hashlib.sha256(CHAMPION.read_bytes()).hexdigest(), hashlib.sha256(CONSENSUS.read_bytes()).hexdigest())
checks['protected_unchanged'] = before == after; checks['mean_abs_move'] = float(np.abs(pred-champion[CLASSES].to_numpy(float)).mean())
print({'file': OUT.name, **checks})
