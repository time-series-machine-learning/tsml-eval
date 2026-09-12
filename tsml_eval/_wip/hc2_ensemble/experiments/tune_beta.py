"""Tune the prior-correction exponent beta on the TRAIN files only.

The combined train posterior is built from the component train probability estimates
(OOB/CV), beta is grid-searched there, and the winner applied to test. No test data is
used, so this is a realisable method.

Also records a fixed beta=0.5 and the per-dataset test-oracle beta as an upper bound.
"""
import os, glob
import numpy as np, pandas as pd

R="D:/Results/UCR"
COMPS={"STC":R+"/ShapeletBased/STC","DrCIF":R+"/IntervalBased/DrCIF",
       "Arsenal":R+"/ConvolutionBased/Arsenal","TDE":R+"/DictionaryBased/TDE"}
N=list(COMPS); NR=30
BETAS=np.round(np.arange(0,1.01,0.125),3)
HERE=os.path.dirname(os.path.abspath(__file__))

def read(p):
    with open(p) as f: head=[next(f) for _ in range(3)]
    nc=int(head[2].split(",")[5])
    a=pd.read_csv(p,skiprows=3,header=None,usecols=list(range(3+nc))).to_numpy()
    return a[:,0].astype(int), a[:,3:3+nc].astype(float)

def acc(y,p): return float((y==p).mean())
def ba(y,p): return float(np.mean([(p[y==c]==c).mean() for c in np.unique(y)]))

def decide(D,prior,beta,seed):
    d=D/(prior**beta) if beta>0 else D.copy()
    s=d.sum(1,keepdims=True); s[s==0]=1; d=d/s
    rng=np.random.RandomState(seed)
    return np.array([int(rng.choice(np.flatnonzero(r==r.max()))) for r in d])

ds=sorted(os.path.basename(x) for x in glob.glob(R+"/Hybrid/HC2/Predictions/*") if os.path.isdir(x))
rows=[]
for i,d in enumerate(ds):
    for r in range(NR):
        try:
            tr={n:read(f"{COMPS[n]}/Predictions/{d}/trainResample{r}.csv") for n in N}
            te={n:read(f"{COMPS[n]}/Predictions/{d}/testResample{r}.csv") for n in N}
        except Exception:
            continue
        ytr=tr[N[0]][0]; yte=te[N[0]][0]
        w=np.array([acc(tr[n][0],np.argmax(tr[n][1],1)) for n in N])**4
        Dtr=np.tensordot(w,np.stack([tr[n][1] for n in N]),axes=1)
        Dte=np.tensordot(w,np.stack([te[n][1] for n in N]),axes=1)
        _,cnt=np.unique(ytr,return_counts=True); prior=cnt/cnt.sum()

        tr_acc={b:acc(ytr,decide(Dtr,prior,b,r)) for b in BETAS}
        tr_ba ={b:ba (ytr,decide(Dtr,prior,b,r)) for b in BETAS}
        te_p  ={b:decide(Dte,prior,b,r) for b in BETAS}

        b_acc=max(BETAS,key=lambda b:(tr_acc[b],-b))    # ties -> smaller beta
        b_ba =max(BETAS,key=lambda b:(tr_ba[b],-b))
        b_or =max(BETAS,key=lambda b:acc(yte,te_p[b]))  # oracle on test accuracy

        rec={"dataset":d,"resample":r,"beta_tuned_acc":b_acc,"beta_tuned_ba":b_ba,"beta_oracle":b_or}
        for name,b in (("hc2",0.0),("fixed50",0.5),("tuned_acc",b_acc),("tuned_ba",b_ba),("oracle",b_or)):
            p=te_p[b]; rec[name+"_acc"]=acc(yte,p); rec[name+"_ba"]=ba(yte,p)
        rows.append(rec)
    print(f"{i+1}/{len(ds)} {d}",flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE,"tune_beta.csv"),index=False)
print("written")
