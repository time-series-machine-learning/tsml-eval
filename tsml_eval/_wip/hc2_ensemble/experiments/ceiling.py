"""Upper bound on what ANY reweighting of the four HC2 components can achieve.

Grid searches the weight simplex per dataset to maximise test balanced accuracy. This
is a cheating oracle and bounds every external compensation scheme.
"""
import os, glob, itertools
import numpy as np, pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

R="D:/Results/UCR"
COMPS={"STC":R+"/ShapeletBased/STC","DrCIF":R+"/IntervalBased/DrCIF",
       "Arsenal":R+"/ConvolutionBased/Arsenal","TDE":R+"/DictionaryBased/TDE"}
N=list(COMPS); HERE=os.path.dirname(os.path.abspath(__file__))

def read(path):
    with open(path) as f: head=[next(f) for _ in range(3)]
    nc=int(head[2].split(",")[5])
    a=pd.read_csv(path,skiprows=3,header=None,usecols=list(range(3+nc))).to_numpy()
    return a[:,0].astype(int), a[:,3:3+nc].astype(float)

def ba(y,p): return float(np.mean([(p[y==c]==c).mean() for c in np.unique(y)]))
def acc(y,p): return float((y==p).mean())

def vote(P,w,seed):
    d=np.tensordot(w,P,axes=1); d=d/d.sum(1,keepdims=True)
    rng=np.random.RandomState(seed)
    return np.array([int(rng.choice(np.flatnonzero(r==r.max()))) for r in d])

# simplex grid, step 1/10
STEP=10
GRID=[np.array(c)/STEP for c in itertools.product(range(STEP+1),repeat=4)
      if sum(c)==STEP and sum(c)>0]
print(f"{len(GRID)} weight vectors")

datasets=sorted(os.path.basename(x) for x in glob.glob(R+"/Hybrid/HC2/Predictions/*") if os.path.isdir(x))
rows=[]
for i,ds in enumerate(datasets):
    r=0
    try:
        tr={n:read(f"{COMPS[n]}/Predictions/{ds}/trainResample{r}.csv") for n in N}
        te={n:read(f"{COMPS[n]}/Predictions/{ds}/testResample{r}.csv") for n in N}
    except Exception as e:
        print("skip",ds,e); continue
    y=te[N[0]][0]; P=np.stack([te[n][1] for n in N])
    w_acc=np.array([acc(tr[n][0],np.argmax(tr[n][1],1))**4 for n in N])
    base=ba(y,vote(P,w_acc,r))
    best=max((ba(y,vote(P,w,r)),tuple(w)) for w in GRID)
    single=max(ba(y,np.argmax(te[n][1],1)) for n in N)
    rows.append({"dataset":ds,"hc2_ba":base,"ceiling_ba":best[0],
                 "best_w":best[1],"best_single_ba":single})
    print(f"{i+1}/{len(datasets)} {ds} base={base:.4f} ceiling={best[0]:.4f}",flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE,"ceiling.csv"),index=False)
print("written")
