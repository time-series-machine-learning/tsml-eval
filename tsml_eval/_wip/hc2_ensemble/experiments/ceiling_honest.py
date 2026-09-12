"""Honest version of the reweighting ceiling.

The per-dataset simplex search picks the best of 286 weight vectors using the test
labels, so its gain is inflated by selection on the evaluation set. Here the weights
are chosen on one stratified half of the test set and scored on the other, both ways,
which is the best any weight-tuning scheme could do given perfect tuning data.
"""
import os, glob, itertools
import numpy as np, pandas as pd

R="D:/Results/UCR"
COMPS={"STC":R+"/ShapeletBased/STC","DrCIF":R+"/IntervalBased/DrCIF",
       "Arsenal":R+"/ConvolutionBased/Arsenal","TDE":R+"/DictionaryBased/TDE"}
N=list(COMPS); HERE=os.path.dirname(os.path.abspath(__file__))

def read(path):
    with open(path) as f: head=[next(f) for _ in range(3)]
    nc=int(head[2].split(",")[5])
    a=pd.read_csv(path,skiprows=3,header=None,usecols=list(range(3+nc))).to_numpy()
    return a[:,0].astype(int), a[:,3:3+nc].astype(float)

def ba(y,p):
    cl=np.unique(y)
    return float(np.mean([(p[y==c]==c).mean() for c in cl])) if len(cl) else np.nan
def acc(y,p): return float((y==p).mean())

def vote(P,w,seed):
    d=np.tensordot(w,P,axes=1); s=d.sum(1,keepdims=True); s[s==0]=1
    d=d/s
    rng=np.random.RandomState(seed)
    return np.array([int(rng.choice(np.flatnonzero(r==r.max()))) for r in d])

STEP=10
GRID=[np.array(c)/STEP for c in itertools.product(range(STEP+1),repeat=4) if sum(c)==STEP]

def halves(y,seed=0):
    rng=np.random.RandomState(seed); a=[]
    for c in np.unique(y):
        idx=np.flatnonzero(y==c); rng.shuffle(idx); a.append(idx[:len(idx)//2])
    A=np.concatenate(a); B=np.setdiff1d(np.arange(len(y)),A)
    return A,B

datasets=sorted(os.path.basename(x) for x in glob.glob(R+"/Hybrid/HC2/Predictions/*") if os.path.isdir(x))
rows=[]
for i,ds in enumerate(datasets):
    r=0
    try:
        tr={n:read(f"{COMPS[n]}/Predictions/{ds}/trainResample{r}.csv") for n in N}
        te={n:read(f"{COMPS[n]}/Predictions/{ds}/testResample{r}.csv") for n in N}
    except Exception as e:
        print("skip",ds,e,flush=True); continue
    y=te[N[0]][0]; P=np.stack([te[n][1] for n in N])
    w_acc=np.array([acc(tr[n][0],np.argmax(tr[n][1],1))**4 for n in N])
    A,B=halves(y)
    if min(len(A),len(B))<len(np.unique(y)): 
        print("skip small",ds,flush=True); continue
    # all grid predictions once
    preds={tuple(w):vote(P,w,r) for w in GRID}
    base=ba(y,vote(P,w_acc,r))
    tuned=[]
    for fit,ev in ((A,B),(B,A)):
        best=max(GRID,key=lambda w: ba(y[fit],preds[tuple(w)][fit]))
        tuned.append(ba(y[ev],preds[tuple(best)][ev]))
    cheat=max(ba(y,preds[tuple(w)]) for w in GRID)
    rows.append({"dataset":ds,"hc2_ba":base,"cheat_ba":cheat,"tuned_ba":float(np.mean(tuned))})
    print(f"{i+1}/{len(datasets)} {ds} base={base:.4f} tuned={rows[-1]['tuned_ba']:.4f} cheat={cheat:.4f}",flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE,"ceiling_honest.csv"),index=False)
print("written")
