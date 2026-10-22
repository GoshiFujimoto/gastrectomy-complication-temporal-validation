import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys; from lib import *
import time
n0,d=load_cohort(); dev=d[d.date<CUTOFF].reset_index(drop=True)
X=dev[ALL_FEAT]; y=dev['G2'].astype(int).values
outer=StratifiedKFold(5,shuffle=True,random_state=SEED)
aucs={m:[] for m in MODELS}; freq={c:0 for c in OPTIONAL}; sel_sets=[]
t0=time.time()
for k,(a,b) in enumerate(outer.split(X,y)):
    pp=Prep(); Xa=pp.fit_transform(X.iloc[a]); Xb=pp.transform(X.iloc[b])
    Xa_df=pd.DataFrame(pp.imp.transform(X.iloc[a]),columns=ALL_FEAT)
    sel=select_features(Xa,y[a],Xa_df,ALL_FEAT,MANDATORY,OPTIONAL,10); sel_sets.append(sel)
    for c in sel:
        if c in freq: freq[c]+=1
    idx=[ALL_FEAT.index(c) for c in sel]
    Xr,yr=resample(Xa[:,idx],y[a])
    for m in MODELS:
        mod=tune(m,Xr,yr,PFAST,10)
        aucs[m].append(roc_auc_score(y[b],mod.predict_proba(Xb[:,idx])[:,1]))
    print(k,sel,{m:round(aucs[m][-1],3) for m in MODELS},round(time.time()-t0),flush=True)
res={'aucs':aucs,'mean':{m:float(np.mean(v)) for m,v in aucs.items()},'freq':freq,'sel':sel_sets}
print(res['mean']); pickle.dump(res,open('nested_res.pkl','wb'))
