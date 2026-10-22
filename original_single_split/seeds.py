import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys
from lib import *
seeds=[int(s) for s in sys.argv[1:]]
n0,d=load_cohort(); dev=d[d.date<CUTOFF].reset_index(drop=True); tv=d[d.date>=CUTOFF].reset_index(drop=True)
R=pickle.load(open('main_res.pkl','rb')); sel=R['sel']; y_dev=dev.G2.astype(int).values; y_tv=tv.G2.astype(int).values
out={}
for sd in seeds:
    itr,ica=train_test_split(np.arange(len(dev)),test_size=0.2,stratify=y_dev,random_state=sd)
    pp=Prep().fit(dev.loc[itr,sel]); Xtr=pp.transform(dev.loc[itr,sel]); Xca=pp.transform(dev.loc[ica,sel]); Xtv=pp.transform(tv[sel])
    Xr,yr=resample(Xtr,y_dev[itr],seed=sd); row={'n_res':len(yr)}
    for m in MODELS:
        mod=tune(m,Xr,yr,PFULL,20,seed=sd); p=Platt(mod).fit(Xca,y_dev[ica]).predict(Xtv)
        row[m]=(roc_auc_score(y_tv,p),)+cal_metrics(y_tv,p)[:1]
    # CCI-only
    pc=Prep().fit(dev.loc[itr,['CCIa']]); lr=LogisticRegression().fit(pc.transform(dev.loc[itr,['CCIa']]),y_dev[itr]); p=Platt(lr).fit(pc.transform(dev.loc[ica,['CCIa']]),y_dev[ica]).predict(pc.transform(tv[['CCIa']])); row['CCI']=(roc_auc_score(y_tv,p),)
    out[sd]=row; print(sd,{k:(round(v[0],3) if isinstance(v,tuple) else v) for k,v in row.items()},flush=True)
    pickle.dump(out,open(f'seeds_{seeds[0]}.pkl','wb'))
