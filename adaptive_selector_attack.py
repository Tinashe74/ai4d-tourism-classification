from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from reproducibility_recovery import data, cross, norm, fit_vec, apply

ROOT=Path(__file__).resolve().parent
CLASSES=["High Cost","Higher Cost","Highest Cost","Low Cost","Lower Cost","Normal Cost"]

def trip_components():
    c,br,hy=data(); champ=__import__('ordinal_attack').champion_oof(); h=np.load(ROOT/'hierarchical_candidate_oof.npy'); l=np.load(ROOT/'local_similarity_oof.npy')
    z=np.log(np.clip(.89*champ+.02*h+.09*l,1e-15,1))/.98; z-=z.max(1,keepdims=True); base=np.exp(z); base/=base.sum(1,keepdims=True)
    st=np.mean([np.load(ROOT/f) for f in ['xgb_compact_struct_oof_seed2026.npy','xgb_compact_struct_oof_seed42.npy','xgb_compact_struct_oof_seed3407.npy']],axis=0); trip=norm(.85*base+.15*st)
    return br,trip,hy

def make_raw_features(tr,te):
    cols=[c for c in te.columns if c!='Tour_ID']; cats=[c for c in cols if tr[c].dtype==object]; nums=[c for c in cols if c not in cats]
    pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cats),('num',SimpleImputer(strategy='median'),nums)])
    return pre.fit_transform(pd.concat([tr[cols],te[cols]],ignore_index=True))

def pred_features(P):
    fs=[]
    for p in P:
        order=np.sort(p,axis=1)[:,::-1]; ent=-(p*np.log(np.clip(p,1e-12,1))).sum(1)[:,None]
        fs.extend([p,ent,order[:,0:1]-order[:,1:2],order[:,0:1]])
    for j in range(1,len(P)): fs.append(P[j]-P[0])
    return np.hstack(fs)

def gate_oof(P,y,seed,kind,floor):
    X=pred_features(P); out=np.zeros((len(y),6)); weights=[]; fold_scores=[]
    losses=np.stack([-np.log(np.clip(P[j][np.arange(len(y)),y],1e-12,1)) for j in range(len(P))],axis=1)
    winner=losses.argmin(1)
    for tr,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y):
        if kind=='logit': model=LogisticRegression(C=.1,max_iter=120,random_state=seed)
        else: model=DecisionTreeClassifier(max_depth=3,min_samples_leaf=150,random_state=seed)
        model.fit(X[tr],winner[tr]); gw=model.predict_proba(X[va]); full=np.zeros((len(va),len(P))); full[:,0]=floor; full+=(1-floor)*gw
        q=np.einsum('nm,mnc->nc',full,np.stack([P[j][va] for j in range(len(P))])); out[va]=norm(q); weights.append(full); fold_scores.append(log_loss(y[va],out[va],labels=np.arange(6)))
    return out,np.vstack(weights),fold_scores,winner

def main():
    tr=pd.read_csv(ROOT/'Train.csv'); te=pd.read_csv(ROOT/'Test.csv'); y,br,hy=data(); base=br
    # Existing calibrated stack B, trip-structure C, and hybrid D.
    cb,_=cross(br,y,2026); ch,_=cross(hy,y,2026); bcal=norm(.85*cb+.15*ch); _,trip,hy=trip_components(); P=np.stack([br,bcal,trip,hy])
    print('BASES',[(i,log_loss(y,P[i],labels=np.arange(6))) for i in range(4)])
    best=None
    for seed in [2026]:
        for kind in ['logit','tree']:
            for floor in [.8,.7,.6]:
                q,w,fs,win=gate_oof(P,y,seed,kind,floor); ll=log_loss(y,q,labels=np.arange(6)); imp=log_loss(y,base,labels=np.arange(6))-ll
                print('GATE',seed,kind,floor,'ll',ll,'imp',imp,'folds',fs,'weights',w.mean(0).round(4).tolist())
                if seed==2026 and (best is None or imp>best[0]): best=(imp,kind,floor,q,w,fs)
    print('BEST_PRIMARY',best[0:3],best[5])
    # Only a robust same-method gain qualifies; no candidate is written otherwise.
    chosen=best; seed_imps=[]
    for seed in [2025,2026,42,777,3407]:
        q,w,fs,_=gate_oof(P,y,seed,chosen[1],chosen[2]); seed_imps.append(log_loss(y,base,labels=np.arange(6))-log_loss(y,q,labels=np.arange(6)))
    print('SEED_IMPS',seed_imps,'mean',np.mean(seed_imps),'min',np.min(seed_imps))
    if chosen[0]<.0004 or np.mean(seed_imps)<.0004 or np.min(seed_imps)<=0:
        print('NO_ROBUST_ADAPTIVE_MODEL_SELECTOR'); return
    print('QUALIFIED_BUT_TEST_BUILD_NOT_IMPLEMENTED')

if __name__=='__main__': main()
