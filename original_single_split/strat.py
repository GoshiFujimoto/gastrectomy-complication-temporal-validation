import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lib import *
import time
n0,d=load_cohort(); R=pickle.load(open('main_res.pkl','rb')); sel=R['sel']
cfg={'open':(d.open==1,pd.Timestamp('2017-04-01'),[c for c in sel if c!='open']),'MIS':(d.open==0,pd.Timestamp('2022-04-01'),MANDATORY)}
out={}
for k,(mask,cut,feats) in cfg.items():
    s=d[mask].reset_index(drop=True); dv=s[s.date<cut].reset_index(drop=True); t=s[s.date>=cut].reset_index(drop=True)
    yd=dv.G2.astype(int).values; yt=t.G2.astype(int).values
    print(k,'dev',len(dv),dv.G2.sum(),'EPV',round(yd.sum()/len(feats),1),'tv',len(t),yt.sum(),round(yt.mean(),3),flush=True)
    a,b=train_test_split(np.arange(len(dv)),test_size=0.2,stratify=yd,random_state=SEED)
    pp=Prep().fit(dv.loc[a,feats]); Xa=pp.transform(dv.loc[a,feats]); Xb=pp.transform(dv.loc[b,feats]); Xt=pp.transform(t[feats])
    Xr,yr=resample(Xa,yd[a]); rows={}; pr={}
    for m in ['LR','RF','XGB','LGBM','CAT']:
        mod=tune(m,Xr,yr,PFULL,20); pl=Platt(mod).fit(Xb,yd[b]); p=pl.predict(Xt); pr[m]=p
        s_,i_,b_,oe=cal_metrics(yt,p); lo,hi=boot_ci(yt,p,roc_auc_score)
        rows[m]={'auc':roc_auc_score(yt,p),'lo':lo,'hi':hi,'slope':s_,'brier':b_}
        print(k,m,{q:round(v,3) for q,v in rows[m].items()},flush=True)
    out[k]={'rows':rows,'probs':pr,'y':yt,'n_dev':len(dv),'n_tv':len(t),'ev_rate':float(yt.mean()),'epv':yd.sum()/len(feats)}
pickle.dump(out,open('strat_res.pkl','wb'))
