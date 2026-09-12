"""Train-BA weighting and PARTIAL prior correction, judged on accuracy.

Partial correction divides by prior ** beta: beta=0 is standard HC2, beta=1 is full
prior correction. If HC2's posterior is miscalibrated toward the majority class, some
beta > 0 could improve accuracy too, not just balanced accuracy.

Run over 30 resamples so the small effects can actually be tested.
"""
import os, glob, sys
import numpy as np, pandas as pd

R="D:/Results/UCR"
COMPS={"STC":R+"/ShapeletBased/STC","DrCIF":R+"/IntervalBased/DrCIF",
       "Arsenal":R+"/ConvolutionBased/Arsenal","TDE":R+"/DictionaryBased/TDE"}
N=list(COMPS); HERE=os.path.dirname(os.path.abspath(__file__))
NR=30
BETAS=[0.25,0.5,0.75,1.0]

def read(path):
    with open(path) as f: head=[next(f) for _ in range(3)]
    nc=int(head[2].split(",")[5])
    a=pd.read_csv(path,skiprows=3,header=None,usecols=list(range(3+nc))).to_numpy()
    return a[:,0].astype(int), a[:,3:3+nc].astype(float)

def acc(y,p): return float((y==p).mean())
def ba(y,p): return float(np.mean([(p[y==c]==c).mean() for c in np.unique(y)]))

def vote(P,w,seed,prior=None,beta=0.0):
    d=np.tensordot(w,P,axes=1)
    if beta>0: d=d/(prior**beta)
    s=d.sum(1,keepdims=True); s[s==0]=1; d=d/s
    rng=np.random.RandomState(seed)
    return np.array([int(rng.choice(np.flatnonzero(r==r.max()))) for r in d])

datasets=sorted(os.path.basename(x) for x in glob.glob(R+"/Hybrid/HC2/Predictions/*") if os.path.isdir(x))
if len(sys.argv)>1: datasets=sys.argv[1:]
rows=[]
for i,ds in enumerate(datasets):
    for r in range(NR):
        try:
            tr={n:read(f"{COMPS[n]}/Predictions/{ds}/trainResample{r}.csv") for n in N}
            te={n:read(f"{COMPS[n]}/Predictions/{ds}/testResample{r}.csv") for n in N}
        except Exception:
            continue
        y=te[N[0]][0]; P=np.stack([te[n][1] for n in N])
        w_acc=np.array([acc(tr[n][0],np.argmax(tr[n][1],1)) for n in N])**4
        w_ba=np.array([ba(tr[n][0],np.argmax(tr[n][1],1)) for n in N])**4
        _,cnt=np.unique(tr[N[0]][0],return_counts=True); prior=cnt/cnt.sum()
        rec={"dataset":ds,"resample":r}
        for name,w in (("HC2",w_acc),("trainba",w_ba)):
            p=vote(P,w,r); rec[name+"_acc"]=acc(y,p); rec[name+"_ba"]=ba(y,p)
        for b in BETAS:
            p=vote(P,w_acc,r,prior,b); rec[f"b{b}_acc"]=acc(y,p); rec[f"b{b}_ba"]=ba(y,p)
            p=vote(P,w_ba,r,prior,b); rec[f"tb{b}_acc"]=acc(y,p); rec[f"tb{b}_ba"]=ba(y,p)
        rows.append(rec)
    print(f"{i+1}/{len(datasets)} {ds}",flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE,"beta.csv"),index=False)
print("written")
