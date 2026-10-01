from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from ordinal_attack import champion_oof

ROOT=Path(__file__).resolve().parent
CLASSES=["High Cost","Higher Cost","Highest Cost","Low Cost","Lower Cost","Normal Cost"]

def sm(z):
    z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
def norm(p):
    p=np.clip(np.asarray(p,dtype=float),1e-12,None); return p/p.sum(1,keepdims=True)
def fit_vec(x,y,lam=.1):
    k=6; init=np.r_[np.ones(k),np.zeros(k)]
    def fg(t):
        a,b=t[:k],t[k:]; p=sm(x*a+b); e=p.copy(); e[np.arange(len(y)),y]-=1
        v=-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean()+lam*((a-1)**2).sum()+lam*(b**2).sum()
        g=np.r_[(e*x).mean(0)+2*lam*(a-1),e.mean(0)+2*lam*b]; return v,g
    return minimize(lambda t:fg(t),init,jac=True,method="L-BFGS-B",options={"maxiter":200,"ftol":1e-10}).x
def apply(p,t): return sm(np.log(np.clip(p,1e-12,1))*t[:6]+t[6:])
def data():
    y=pd.factorize(pd.read_csv(ROOT/"Train.csv").cost_category,sort=True)[0]
    c=champion_oof(); h=np.load(ROOT/"hierarchical_candidate_oof.npy"); l=np.load(ROOT/"local_similarity_oof.npy")
    z=np.log(np.clip(.89*c+.02*h+.09*l,1e-15,1))/.98; base=sm(z)
    raw=np.mean([np.load(ROOT/f) for f in ["xgb_compact_oof.npy","xgb_compact_oof_seed42.npy","xgb_compact_oof_seed3407.npy"]],axis=0)
    br=norm(.9*base+.1*raw)
    hy=np.mean([np.load(ROOT/f) for f in ["xgb_compact_hybrid_oof_seed2026.npy","xgb_compact_hybrid_oof_seed42.npy","xgb_compact_hybrid_oof_seed777.npy","xgb_compact_hybrid_oof_seed3407.npy"]],axis=0); hy=norm(hy)
    return y,br,hy
def cross(p,y,seed):
    out=np.zeros_like(p); fs=[]
    for tr,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(p,y):
        t=fit_vec(np.log(np.clip(p[tr],1e-12,1)),y[tr]); out[va]=apply(p[va],t); fs.append(log_loss(y[va],out[va],labels=np.arange(6)))
    return out,fs
def main():
    y,br,hy=data(); base=log_loss(y,br,labels=np.arange(6)); print("BASE",base)
    results=[]
    for seed in [2025,2026,42,777,3407]:
        cb,fb=cross(br,y,seed); ch,fh=cross(hy,y,seed); p=norm(.85*cb+.15*ch); ll=log_loss(y,p,labels=np.arange(6)); results.append((seed,ll,base-ll,fb,fh)); print("SEED",seed,"OOF",ll,"IMP",base-ll,"folds",[round(x,8) for x in fb],[round(x,8) for x in fh])
    imps=np.array([r[2] for r in results]); print("MEAN",imps.mean(),"MIN",imps.min(),"POS",int((imps>0).sum()))
    if imps.mean()<.0004 or (imps>0).sum()<4:
        print("HOLD_REPRODUCIBILITY_NOT_ESTABLISHED"); return
    # Exact original test-time procedure: protected breakthrough file is br_test.
    champ=pd.read_csv(ROOT/"candidate_xgboost_breakthrough_v3.csv"); brt=champ[CLASSES].to_numpy(float)
    hyt=np.mean([np.load(ROOT/f) for f in ["xgb_compact_hybrid_test_seed2026.npy","xgb_compact_hybrid_test_seed42.npy","xgb_compact_hybrid_test_seed777.npy","xgb_compact_hybrid_test_seed3407.npy"]],axis=0)
    tbr=fit_vec(np.log(np.clip(br,1e-12,1)),y); thy=fit_vec(np.log(np.clip(hy,1e-12,1)),y); pred=norm(.85*apply(brt,tbr)+.15*apply(hyt,thy))
    sample=pd.read_csv(ROOT/"SampleSubmission.csv"); out=sample.copy(); out[CLASSES]=pred
    if list(out.columns)!=list(sample.columns) or len(out)!=6169 or not out.iloc[:,0].equals(sample.iloc[:,0]): print("INTEGRITY_FAILED"); return
    out.to_csv(ROOT/"candidate_final_calibrated_xgb_reproducible_v2.csv",index=False)
    p=out[CLASSES].to_numpy(float); print("CREATED",len(out),"ID",out.iloc[:,0].name,"finite",np.isfinite(p).all(),"range",((p>=0)&(p<=1)).all(),"sumerr",np.max(np.abs(p.sum(1)-1)),"dup",out.iloc[:,0].duplicated().sum())
if __name__=="__main__": main()
