import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from xgb_compact_oof import prepare
from ordinal_attack import champion_oof

ROOT=Path(__file__).resolve().parent
tr=pd.read_csv(ROOT/'Train.csv'); te=pd.read_csv(ROOT/'Test.csv'); y=pd.factorize(tr.cost_category,sort=True)[0]
x,xt=prepare(tr,te)
c=champion_oof(); h=np.load(ROOT/'hierarchical_candidate_oof.npy'); l=np.load(ROOT/'local_similarity_oof.npy')
z=np.log(np.clip(.89*c+.02*h+.09*l,1e-15,1))/.98; z-=z.max(1,keepdims=True); base=np.exp(z); base/=base.sum(1,keepdims=True)
raw=np.mean([np.load(ROOT/f) for f in ['xgb_compact_oof.npy','xgb_compact_oof_seed42.npy','xgb_compact_oof_seed3407.npy']],axis=0)
global_p=.9*base+.1*raw; global_p/=global_p.sum(1,keepdims=True); baseline=log_loss(y,global_p)

nights=tr.night_mainland.fillna(0)+tr.night_zanzibar.fillna(0)
pkg=tr[[c for c in tr.columns if c.startswith('package_')]].eq('Yes').sum(axis=1)
band=pd.cut(nights,[-1,0,3,7,14,30,999],labels=False).fillna(-1).astype(int)
regimes={
 'long_no_package': (band>=5)&(pkg==0),
 'south_africa_package': (tr.country=='SOUTH AFRICA')&(tr.tour_arrangement=='Package Tour'),
 'business_conference': tr.purpose.isin(['Business','Meetings and Conference']),
}
print('baseline',baseline)
for n,m in regimes.items(): print('regime',n,'rows',int(m.sum()))

folds=StratifiedKFold(5,shuffle=True,random_state=2026)
spec={n:np.full((len(tr),6),np.nan) for n in regimes}
for fold,(a,b) in enumerate(folds.split(x,y),1):
    for name,mask in regimes.items():
        atr=a[mask.to_numpy()[a]]; vmask=mask.to_numpy()[b]
        if len(atr)<150: continue
        model=XGBClassifier(n_estimators=260,max_depth=3,learning_rate=.055,min_child_weight=8,subsample=.9,colsample_bytree=.9,reg_alpha=.2,reg_lambda=10,objective='multi:softprob',num_class=6,eval_metric='mlogloss',tree_method='hist',n_jobs=2,random_state=2026+fold,early_stopping_rounds=30)
        # Specialist is trained only on regime rows in the training fold.
        model.fit(x.iloc[atr],y[atr],eval_set=[(x.iloc[atr],y[atr])],verbose=False)
        rows=b[vmask]
        if len(rows): spec[name][rows]=model.predict_proba(x.iloc[rows])
        print('fold',fold,'specialist',name,'train',len(atr),'valid',len(rows),'best',model.best_iteration,flush=True)

def evaluate(p): return log_loss(y,p)
best=[]
for name,s in spec.items():
    valid=np.isfinite(s[:,0]);
    for gate_name,gate in [('membership',valid),('confident',valid&(global_p.max(1)>=.6)),('disagreement',valid&(np.argmax(global_p,1)!=np.argmax(s,1)))]:
        for w in [.01,.02,.03,.05,.075,.10,.15]:
            q=global_p.copy(); q[gate]=(1-w)*q[gate]+w*s[gate]; q/=q.sum(1,keepdims=True); score=evaluate(q)
            best.append((score,baseline-score,name,gate_name,w,int(gate.sum())))
for combo in [('long_no_package','south_africa_package'),('long_no_package','business_conference'),('south_africa_package','business_conference'),('long_no_package','south_africa_package','business_conference')]:
    for w in [.02,.05,.10]:
        q=global_p.copy(); active=np.zeros(len(y),bool)
        for name in combo:
            s=spec[name]; g=np.isfinite(s[:,0]); active|=g; q[g]=(1-w)*q[g]+w*s[g]
        q/=q.sum(1,keepdims=True); best.append((evaluate(q),baseline-evaluate(q),'+'.join(combo),'membership',w,int(active.sum())))
best=sorted(best)
print('TOP_RESULTS')
for r in best[:20]: print(r)
np.save(ROOT/'xgb_moe_specialist_predictions.npy',np.stack([np.nan_to_num(spec[n],nan=0) for n in regimes]))
pd.DataFrame(best,columns=['oof','improvement','specialists','gate','weight','rows']).to_csv(ROOT/'xgb_moe_results.csv',index=False)
