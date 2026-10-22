import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lib import *
import time, sys
RESAMPLE = (sys.argv[1] != 'nores') if len(sys.argv)>1 else True
tag = '' if RESAMPLE else '_nores'
n0,d=load_cohort(); 
dev=d[d.date<CUTOFF].reset_index(drop=True); tv=d[d.date>=CUTOFF].reset_index(drop=True)
y_dev=dev['G2'].astype(int).values; y_tv=tv['G2'].astype(int).values
itr,ica=train_test_split(np.arange(len(dev)),test_size=0.2,stratify=y_dev,random_state=SEED)
# final predictor set re-derived on full development cohort
pp_dev=Prep(); Xd=pp_dev.fit_transform(dev[ALL_FEAT]); Xd_df=pd.DataFrame(pp_dev.imp.transform(dev[ALL_FEAT]),columns=ALL_FEAT)
sel=select_features(Xd,y_dev,Xd_df,ALL_FEAT,MANDATORY,OPTIONAL,10)
print('selected',len(sel),sel,'EPV train',y_dev[itr].sum()/len(sel),'EPV dev',y_dev.sum()/len(sel))
# preprocessing fitted on training only
pp=Prep().fit(dev.loc[itr,sel]); Xtr=pp.transform(dev.loc[itr,sel]); Xca=pp.transform(dev.loc[ica,sel]); Xtv=pp.transform(tv[sel])
ytr,yca=y_dev[itr],y_dev[ica]
if RESAMPLE: Xr,yr=resample(Xtr,ytr)
else: Xr,yr=Xtr,ytr
print('train',len(ytr),'->',len(yr),'pos',ytr.sum(),'->',int(yr.sum()),flush=True)
mods={}; t0=time.time()
for m in MODELS:
    mods[m]=tune(m,Xr,yr,PFULL,20); print(m,'tuned',round(time.time()-t0),mods[m].get_params() if m in('LR',) else '',flush=True)
from sklearn.base import clone
stack=StackingClassifier(estimators=[(m.lower(),clone(mods[m])) for m in MODELS],final_estimator=LogisticRegression(max_iter=1000),cv=StratifiedKFold(3,shuffle=True,random_state=SEED),n_jobs=1)
stack.fit(Xr,yr); mods['Stack']=stack
P={}; rows=[]
for m,mod in mods.items():
    pl=Platt(mod).fit(Xca,yca)
    P[m]={'train':pl.predict(Xtr),'cal':pl.predict(Xca),'tv':pl.predict(Xtv),'raw_tv':mod.predict_proba(Xtv)[:,1],'platt':pl}
    for sp,(Xs,ys) in {'train':(Xtr,ytr),'cal':(Xca,yca),'tv':(Xtv,y_tv)}.items():
        p=P[m][sp]; s,i,b,oe=cal_metrics(ys,p)
        r={'Model':m,'Split':sp,'AUC':roc_auc_score(ys,p),'PR':average_precision_score(ys,p),'Brier':b,'Slope':s,'Int':i,'OE':oe}
        if sp=='tv':
            r['AUC_lo'],r['AUC_hi']=boot_ci(ys,p,roc_auc_score); r['PR_lo'],r['PR_hi']=boot_ci(ys,p,average_precision_score)
        rows.append(r)
res=pd.DataFrame(rows); print(res[res.Split=='tv'].round(3).to_string()); print(res[res.Split=='train'][['Model','AUC']].round(3).to_string())
pickle.dump({'sel':sel,'dev':dev,'tv':tv,'itr':itr,'ica':ica,'Xtr':Xtr,'Xca':Xca,'Xtv':Xtv,'ytr':ytr,'yca':yca,'y_tv':y_tv,'Xr':Xr,'yr':yr,'P':P,'res':res,'mods':mods,'pp':pp},open(f'main_res{tag}.pkl','wb'))
