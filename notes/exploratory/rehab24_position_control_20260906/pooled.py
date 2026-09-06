import csv, collections, statistics
rows=list(csv.DictReader(open('data/REHAB24-6/Segmentation.csv'),delimiter=';'))
def midranks(v):
    idx=sorted(range(len(v)),key=lambda i:v[i]); r=[0.0]*len(v); i=0
    while i<len(idx):
        j=i
        while j+1<len(idx) and v[idx[j+1]]==v[idx[i]]: j+=1
        m=(i+j)/2+1
        for k in range(i,j+1): r[idx[k]]=m
        i=j+1
    return r
def auc(lab,sc):
    P=sum(lab); N=len(lab)-P
    if P==0 or N==0: return None
    r=midranks(sc)
    return (sum(r[i] for i in range(len(lab)) if lab[i]==1)-P*(P+1)/2)/(P*N)
def best_ba(lab,sc):
    P=sum(lab); N=len(lab)-P; best=0
    for t in sorted(set(sc)):
        tp=sum(1 for l,s in zip(lab,sc) if s< t and l==1)   # small index -> predict correct
        tn=sum(1 for l,s in zip(lab,sc) if s>=t and l==0)
        best=max(best,(tp/P+tn/N)/2)
    return best
for tag,sel in [("all subjects",lambda r:True),("P1-P9",lambda r:r['person_id']!='10')]:
    rs=[r for r in rows if sel(r)]
    lab=[int(r['correctness']) for r in rs]; pos=[float(r['repetition_number']) for r in rs]
    frac=[0.0]*len(rs)
    L=collections.Counter((r['exercise_id'],r['video_id']) for r in rs)
    for i,r in enumerate(rs): frac[i]=float(r['repetition_number'])/L[(r['exercise_id'],r['video_id'])]
    print(f"--- {tag}: n_reps={len(rs)}  base_rate={sum(lab)/len(lab):.3f}")
    print(f"    pooled AUC (raw rep_number)      = {auc(lab,pos):.4f}   flipped {1-auc(lab,pos):.4f}")
    print(f"    pooled AUC (rep_number/session_n)= {auc(lab,frac):.4f}   flipped {1-auc(lab,frac):.4f}")
    print(f"    best-threshold BA (raw, OPTIMISTIC) = {best_ba(lab,pos):.4f}")
    print(f"    best-threshold BA (frac, OPTIMISTIC)= {best_ba(lab,frac):.4f}")
