from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parent
tr = pd.read_csv(ROOT / 'Train.csv')
te = pd.read_csv(ROOT / 'Test.csv')
y = pd.factorize(tr.cost_category, sort=True)[0]
files = ['geo_frequency', 'onehot', 'interaction', 'catboost', 'joint_frequency', 'joint_frequency_seed2024', 'joint_frequency_seed3407']
champion_arrays = np.stack([np.load(ROOT / (f + '_oof.npy')) for f in files])
champion_weights = np.array([.0413259998, .2217763817, .1500358034, .1044899380, .0678346740, .1734401381, .2410970651])
c = np.einsum('m,mnc->nc', champion_weights, champion_arrays)
c = np.log(np.clip(c, 1e-15, 1)) / .9693158908
c -= c.max(1, keepdims=True); c = np.exp(c); c /= c.sum(1, keepdims=True)
correction = np.mean([np.load(ROOT / f) for f in ['consensus_correction_oof.npy', 'consensus_correction_alt_oof.npy', 'consensus_correction_seed42_oof.npy', 'consensus_correction_seed777_oof.npy']], axis=0)
c = .7 * c + .3 * correction
h = np.load(ROOT / 'hierarchical_candidate_oof.npy')
l = np.load(ROOT / 'local_similarity_oof.npy')
z = np.log(np.clip(.89*c + .02*h + .09*l, 1e-15, 1)) / .98
z -= z.max(1, keepdims=True)
base = np.exp(z); base /= base.sum(1, keepdims=True)
raw = np.mean([np.load(ROOT / f) for f in ['xgb_compact_oof.npy', 'xgb_compact_oof_seed42.npy', 'xgb_compact_oof_seed3407.npy']], axis=0)
global_p = .9 * base + .1 * raw; global_p /= global_p.sum(1, keepdims=True)
baseline = log_loss(y, global_p)

masks = {
    'sa': ((tr.country == 'SOUTH AFRICA') & (tr.tour_arrangement == 'Package Tour')).to_numpy(),
    'biz': tr.purpose.isin(['Business', 'Meetings and Conference']).to_numpy(),
}
stack = np.load(ROOT / 'xgb_moe_specialist_predictions.npy')
spec = {'sa': stack[1], 'biz': stack[2]}
weights = [.02, .03, .05, .075, .10, .125, .15, .175, .20]
rows = []
for ws in weights:
    for wb in weights:
        p = global_p.copy()
        p[masks['sa']] = (1-ws) * p[masks['sa']] + ws * spec['sa'][masks['sa']]
        p[masks['biz']] = (1-wb) * p[masks['biz']] + wb * spec['biz'][masks['biz']]
        p /= p.sum(1, keepdims=True)
        score = log_loss(y, p)
        rows.append((score, baseline-score, ws, wb, int(masks['sa'].sum()), int(masks['biz'].sum())))
out = pd.DataFrame(rows, columns=['oof','improvement','w_sa','w_biz','sa_rows','biz_rows']).sort_values('oof')
out.to_csv(ROOT / 'xgb_moe_weight_grid.csv', index=False)
print('baseline', repr(baseline))
print(out.head(15).to_string(index=False))
