import csv, collections, json, os, numpy as np
BASE='data/REHAB24-6/processed/videomae_identity_control'; SEEDS=(42,7,1234)
per=collections.defaultdict(lambda: collections.defaultdict(list)); meta={}
for s in SEEDS:
    for r in csv.DictReader(open(f'{BASE}/oof_seed{s}.csv')):
        k=(r['exercise_id'],r['video_id'],int(r['repetition_number']))
        per[k][s].append(float(r['probability'])); meta[k]=(r['person_id'],int(r['label']))
S=collections.defaultdict(list)
for k,d in per.items():
    pid,lab=meta[k]
    if pid=='10': continue
    S[(k[0],k[1],pid)].append((k[2],lab,[float(np.mean(d[s])) for s in SEEDS]))
sess=[]
for k,items in S.items():
    items.sort(); L=np.array([i[1] for i in items],float)
    if len(set(L))<2: continue
    sess.append((k[2],np.array([i[0] for i in items],float),L,
                 np.array([[i[2][j] for i in items] for j in range(len(SEEDS))],float)))
def cm(sc): return (sc[:,None]>sc[None,:]).astype(float)+0.5*(sc[:,None]==sc[None,:])
CM=[np.mean([cm(sc) for sc in s[3]],axis=0) for s in sess]
PM=[cm(-s[1]) for s in sess]
def stat(labels,C,maxd=None):
    """Position-balanced within-session AUC: each session's correct-first and
    incorrect-first pair halves are weighted to contribute 0.5 each, so the
    position rule scores exactly 0.5000 by construction."""
    ps=collections.defaultdict(list); ns=0
    for (pid,pos,_,_),Ci,L in zip(sess,C,labels):
        d=pos[:,None]-pos[None,:]
        keep=np.ones_like(d,float) if maxd is None else (np.abs(d)<=maxd).astype(float)
        cf=keep*(d<0); wf=keep*(d>0)
        n1=float(L@cf@(1-L)); n2=float(L@wf@(1-L))
        if n1==0 or n2==0: continue
        ps[pid].append(0.5*float(L@(cf*Ci)@(1-L))/n1 + 0.5*float(L@(wf*Ci)@(1-L))/n2); ns+=1
    v=[np.mean(x) for x in ps.values()]
    return float(np.mean(v)), len(v), ns
rng=np.random.default_rng(20260906); N=10000; out=[]
for tag,md in [("position-balanced (all pairs)",None),("position-balanced (|d|<=3)",3),("position-balanced (|d|=1)",1)]:
    obs=stat([s[2] for s in sess],CM,md); pos=stat([s[2] for s in sess],PM,md)
    P=[rng.permuted(np.tile(s[2],(N,1)),axis=1) for s in sess]
    nl=np.array([stat([q[p] for q in P],CM,md)[0] for p in range(N)])
    pg=(1+int((nl>=obs[0]).sum()))/(N+1)
    print(f"{tag:30s} model={obs[0]:.4f}  position={pos[0]:.4f}  subjects={obs[1]}  sessions={obs[2]}  null={nl.mean():.4f}+/-{nl.std():.4f}  p={pg:.5f}")
    out.append(dict(arm=tag,model=obs[0],position_rule=pos[0],n_subjects=obs[1],n_sessions=obs[2],null_mean=float(nl.mean()),null_sd=float(nl.std()),p_greater=pg,n_perm=N,perm_seed=20260906))
json.dump(out,open(os.path.join(os.path.dirname(__file__),'position_balanced_result.json'),'w'),indent=2)
print("saved position_balanced_result.json")

# Same-subset comparison: unrestricted within-session AUC on ONLY the 34 sessions
# where the balanced statistic is computable, so the drop is attributable to
# balancing rather than to the change of session subset.
def plain(labels,C,subset):
    ps=collections.defaultdict(list)
    for (pid,pos,_,_),Ci,L,ok in zip(sess,C,labels,subset):
        if not ok: continue
        d=pos[:,None]-pos[None,:]; M=np.ones_like(d,float)
        ps[pid].append(float(L@(M*Ci)@(1-L))/float(L@M@(1-L)))
    v=[np.mean(x) for x in ps.values()]; return float(np.mean(v)), len(v), sum(subset)
subset=[]
for (pid,pos,L,_) in sess:
    d=pos[:,None]-pos[None,:]
    subset.append(float(L@(d<0)@(1-L))>0 and float(L@(d>0)@(1-L))>0)
m34=plain([s[2] for s in sess],CM,subset); p34=plain([s[2] for s in sess],PM,subset)
mAll=plain([s[2] for s in sess],CM,[True]*len(sess)); pAll=plain([s[2] for s in sess],PM,[True]*len(sess))
print(f"unrestricted, all 61 sessions   model={mAll[0]:.4f}  position={pAll[0]:.4f}  sessions={mAll[2]}")
print(f"unrestricted, the same 34       model={m34[0]:.6f}  position={p34[0]:.6f}  sessions={m34[2]}")
