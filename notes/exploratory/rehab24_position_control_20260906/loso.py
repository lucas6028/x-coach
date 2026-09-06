import csv, collections, statistics
rows=[r for r in csv.DictReader(open('data/REHAB24-6/Segmentation.csv'),delimiter=';') if r['person_id']!='10']
def ba(lab,sc,t):
    P=sum(lab); N=len(lab)-P
    if P==0 or N==0: return None
    tp=sum(1 for l,s in zip(lab,sc) if s< t and l==1); tn=sum(1 for l,s in zip(lab,sc) if s>=t and l==0)
    return (tp/P+tn/N)/2
def fit(lab,sc):
    cand=sorted(set(sc)); return max(cand,key=lambda t: ba(lab,sc,t))
FEAT={'rep position (raw index)':lambda r: float(r['repetition_number']),
      'rep position (fraction)' :None,
      'rep duration (frames)'   :lambda r: float(int(r['last_frame'])-int(r['first_frame']))}
L=collections.Counter((r['exercise_id'],r['video_id']) for r in rows)
def frac(r): return float(r['repetition_number'])/L[(r['exercise_id'],r['video_id'])]
FEAT['rep position (fraction)']=frac
subs=sorted({r['person_id'] for r in rows},key=int)
for name,fn in FEAT.items():
    out=[]
    for s in subs:
        tr=[r for r in rows if r['person_id']!=s]; te=[r for r in rows if r['person_id']==s]
        t=fit([int(r['correctness']) for r in tr],[fn(r) for r in tr])
        v=ba([int(r['correctness']) for r in te],[fn(r) for r in te],t)
        if v is not None: out.append(v)
    print(f"{name:26s} LOSO subject-macro BA = {statistics.mean(out):.6f} +/- {statistics.stdev(out):.4f}   ({sum(v>0.5 for v in out)}/{len(out)} > 0.5)")
