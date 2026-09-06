import csv, collections, numpy as np
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
def score(masks,use_pos):
    ps=collections.defaultdict(list); cn=cd=0.0; ns=0
    for (pid,pos,L,SC),M in zip(sess,masks):
        # model: mean over seeds of concordance; position rule: SMALL index = predict correct
        C = cm(-pos) if use_pos else np.mean([cm(sc) for sc in SC],axis=0)
        num=float(L@(M*C)@(1-L)); den=float(L@M@(1-L)); cn+=num; cd+=den
        if den>0: ps[pid].append(num/den); ns+=1
    v=[np.mean(x) for x in ps.values()]
    return (float(np.mean(v)) if v else float('nan')), ns, cd
def band(lo,hi,sign):
    out=[]
    for _,pos,_,_ in sess:
        d=pos[:,None]-pos[None,:]; a=np.abs(d)
        m=((a>=lo)&(a<=hi)).astype(float)
        if sign=='cf': m*= (d<0)   # correct(row) earlier than incorrect(col)
        if sign=='if': m*= (d>0)   # correct LATER -> position points the wrong way
        out.append(m.astype(float))
    return out
print(f"{'pair subset':32s} {'model':>7s} {'position':>9s} {'pairs':>6s} {'sess':>5s}")
rows=[("all pairs",(1,99,'all')),("correct-first pairs",(1,99,'cf')),("incorrect-first pairs",(1,99,'if')),
      ("|d|=1 all",(1,1,'all')),("|d|=1 correct-first",(1,1,'cf')),("|d|=1 incorrect-first",(1,1,'if')),
      ("|d|<=3 correct-first",(1,3,'cf')),("|d|<=3 incorrect-first",(1,3,'if'))]
res={}
for tag,(lo,hi,sg) in rows:
    M=band(lo,hi,sg); m,ns,np_=score(M,False); p,_,_=score(M,True)
    res[tag]=(m,p,np_,ns); print(f"{tag:32s} {m:7.4f} {p:9.4f} {np_:6.0f} {ns:5d}")
for a,b,name in [("correct-first pairs","incorrect-first pairs","position-balanced (all |d|)"),
                 ("|d|=1 correct-first","|d|=1 incorrect-first","position-balanced (|d|=1)"),
                 ("|d|<=3 correct-first","|d|<=3 incorrect-first","position-balanced (|d|<=3)")]:
    print(f"{name:32s} {(res[a][0]+res[b][0])/2:7.4f} {(res[a][1]+res[b][1])/2:9.4f}")
