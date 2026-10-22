from dbl import *
import dbl, statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import kruskal, chi2_contingency
rng=np.random.default_rng(SEED); F={}
# ---------- predictions (primary pipeline) ----------
cfgs={'Ridge':('ridge',ALL_FEAT),'CatBoost':('cat',ALL_FEAT),'CCI':('lr0',['CCIa']),'ASA':('lr0',['ASA-PS'])}
preds={}
for nm,(kind,cols) in cfgs.items():
    if kind=='cat':
        rs=[run(kind,cols,PRIMARY,s) for s in range(5)]; p=np.mean([r[0] for r in rs],0); yy,bb=rs[0][1],rs[0][2]
    else: p,yy,bb=run(kind,cols,PRIMARY)
    preds[nm]=p
idx=np.concatenate([np.where(te)[0] for tr,te in blocks_from(PRIMARY)])
open_=d.open.values[idx]; dates=d.date.values[idx]
F['n_test']=len(yy); F['ev_test']=int(yy.sum()); F['block_n']=[int((bb==k).sum()) for k in range(3)]; F['block_ev']=[int(yy[bb==k].sum()) for k in range(3)]
F['train_only_n']=int((d.date<'2010-04-01').sum()); F['train_only_ev']=int(y[(d.date<'2010-04-01').values].sum())
F['train_sizes']=[int((d.date<b).sum()) for b in ['2010-04-01','2014-04-01','2019-04-01']]; F['train_events']=[int(y[(d.date<b).values].sum()) for b in ['2010-04-01','2014-04-01','2019-04-01']]
# ---------- performance with bootstrap CIs ----------
def boot_all(p,n=2000):
    r=np.random.default_rng(SEED); A=[];PR=[];BR=[];SL=[];IN=[]
    for _ in range(n):
        i=r.integers(0,len(yy),len(yy))
        if len(np.unique(yy[i]))<2: continue
        A.append(roc_auc_score(yy[i],p[i])); PR.append(average_precision_score(yy[i],p[i])); BR.append(brier_score_loss(yy[i],p[i])); s_,i_,_,_=cal_metrics(yy[i],p[i]); SL.append(s_); IN.append(i_)
    q=lambda v: np.percentile(v,[2.5,97.5]).tolist(); return dict(auc_ci=q(A),pr_ci=q(PR),brier_ci=q(BR),slope_ci=q(SL),int_ci=q(IN))
perf={}
for nm,p in preds.items():
    s_,i_,br,oe=cal_metrics(yy,p); perf[nm]=dict(auc=roc_auc_score(yy,p),pr=average_precision_score(yy,p),brier=br,slope=s_,int=i_,oe=oe,**boot_all(p),
        block_auc=[roc_auc_score(yy[bb==k],p[bb==k]) for k in range(3)],block_slope=[cal_metrics(yy[bb==k],p[bb==k])[0] for k in range(3)],block_oe=[cal_metrics(yy[bb==k],p[bb==k])[3] for k in range(3)])
F['perf']=perf
F['null_brier']=brier_score_loss(yy,np.full(len(yy),yy.mean()))
# ---------- comparisons ----------
comp={}
for tgt,refs in [('Ridge',['CCI','ASA']),('CatBoost',['CCI','ASA']),('Ridge',['CatBoost'])]:
    for ref in refs:
        a1,a2,pdl=delong_test(yy,preds[tgt],preds[ref]); r=np.random.default_rng(SEED); da=[];db=[];nr=[];idi=[]
        for _ in range(2000):
            i=r.integers(0,len(yy),len(yy))
            if len(np.unique(yy[i]))<2: continue
            da.append(roc_auc_score(yy[i],preds[tgt][i])-roc_auc_score(yy[i],preds[ref][i])); db.append(brier_score_loss(yy[i],preds[tgt][i])-brier_score_loss(yy[i],preds[ref][i])); a,b=nri_idi(yy[i],preds[tgt][i],preds[ref][i]); nr.append(a); idi.append(b)
        n0_,i0=nri_idi(yy,preds[tgt],preds[ref]); pv=lambda v,e: float(2*(1-st.norm.cdf(abs(e/np.std(v)))))
        q=lambda v: np.percentile(v,[2.5,97.5]).tolist()
        comp[f'{tgt}_vs_{ref}']=dict(dauc=a1-a2,dauc_ci=q(da),p_delong=float(pdl),dbrier=perf[tgt]['brier']-perf[ref]['brier'],dbrier_ci=q(db),nri=n0_,nri_ci=q(nr),p_nri=pv(nr,n0_),idi=i0,idi_ci=q(idi),p_idi=pv(idi,i0))
F['comp']=comp
fam=['Ridge_vs_CCI','Ridge_vs_ASA']; pl=[comp[k]['p_delong'] for k in fam]+[comp[k]['p_nri'] for k in fam]+[comp[k]['p_idi'] for k in fam]
ph=multipletests(pl,method='holm')[1]; F['holm']={'delong_CCI':ph[0],'delong_ASA':ph[1],'nri_CCI':ph[2],'nri_ASA':ph[3],'idi_CCI':ph[4],'idi_ASA':ph[5]}
# ---------- calibration quintiles ----------
def quint(p,y_):
    q=pd.qcut(pd.Series(p).rank(method='first'),5,labels=False).values; out=[]
    for k in range(5):
        m=q==k; n=int(m.sum()); o=float(y_[m].mean()); se=np.sqrt(o*(1-o)/n); out.append(dict(pred=float(p[m].mean()),obs=o,lo=o-1.96*se,hi=o+1.96*se,n=n))
    return out
F['calq']={nm:quint(p,yy) for nm,p in preds.items()}
# ---------- DCA ----------
ths=np.round(np.arange(0.05,0.61,0.01),2)
def nbf(p,y_,t): return ((p>=t)&(y_==1)).sum()/len(y_)-((p>=t)&(y_==0)).sum()/len(y_)*t/(1-t)
F['dca']={'th':ths.tolist(),**{nm:[nbf(p,yy,t) for t in ths] for nm,p in preds.items()},'all':[yy.mean()-(1-yy.mean())*t/(1-t) for t in ths]}
F['dca_ridge_open']=[nbf(preds['Ridge'][open_==1],yy[open_==1],t) for t in ths]; F['dca_ridge_mis']=[nbf(preds['Ridge'][open_==0],yy[open_==0],t) for t in ths]
F['dca_all_open']=[yy[open_==1].mean()-(1-yy[open_==1].mean())*t/(1-t) for t in ths]; F['dca_all_mis']=[yy[open_==0].mean()-(1-yy[open_==0].mean())*t/(1-t) for t in ths]
dd={}
for t in (0.25,0.30,0.35):
    r=np.random.default_rng(SEED); a=[];b=[]
    for _ in range(2000):
        i=r.integers(0,len(yy),len(yy)); a.append(nbf(preds['Ridge'][i],yy[i],t)-nbf(preds['CCI'][i],yy[i],t)); b.append(nbf(preds['Ridge'][i],yy[i],t)-(yy[i].mean()-(1-yy[i].mean())*t/(1-t)))
    dd[t]=dict(nb_ridge=nbf(preds['Ridge'],yy,t),nb_cci=nbf(preds['CCI'],yy,t),nb_all=yy.mean()-(1-yy.mean())*t/(1-t),d_cci_ci=np.percentile(a,[2.5,97.5]).tolist(),d_all_ci=np.percentile(b,[2.5,97.5]).tolist())
F['dca_pts']=dd
# ---------- ROC curves ----------
F['roc']={nm:roc_curve(yy,p) for nm,p in preds.items()}; F['roc_block']={k:roc_curve(yy[bb==k],preds['Ridge'][bb==k]) for k in range(3)}
# ---------- approach-specific ----------
ap={}
for nm in ['Ridge','CatBoost','CCI','ASA']:
    ap[nm]={k:(int((open_==v).sum()),int(yy[open_==v].sum()),roc_auc_score(yy[open_==v],preds[nm][open_==v]),boot_ci(yy[open_==v],preds[nm][open_==v],roc_auc_score,2000)) for k,v in (('open',1),('mis',0))}
F['approach']=ap
r=np.random.default_rng(SEED); v=[]
for _ in range(2000):
    i=r.integers(0,len(yy),len(yy)); o=open_[i]==1
    try: v.append(roc_auc_score(yy[i][o],preds['Ridge'][i][o])-roc_auc_score(yy[i][~o],preds['Ridge'][i][~o]))
    except Exception: pass
F['approach_diff']=dict(diff=ap['Ridge']['open'][2]-ap['Ridge']['mis'][2],ci=np.percentile(v,[2.5,97.5]).tolist())
lp=lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))); L=lp(preds['Ridge'])
f=sm.Logit(yy,sm.add_constant(np.column_stack([L,open_,L*open_]))).fit(disp=0); F['approach_interaction']=dict(slope_mis=float(f.params[1]),slope_open=float(f.params[1]+f.params[3]),p=float(f.pvalues[3]))
F['mis_calib']={k:cal_metrics(yy[open_==v],preds['Ridge'][open_==v]) for k,v in (('open',1),('mis',0))}
# open-by-period
F['open_rate_period']=[float(d.open[(d.date>=a)&(d.date<b)].mean()) for a,b in [('2000-04-01','2010-04-01'),('2010-04-01','2014-04-01'),('2014-04-01','2019-04-01'),('2019-04-01','2025-04-01')]]
# ---------- sensitivity analyses ----------
sens={}
def srow(p,y2,b2): s_=summ(p,y2,b2); return dict(n=s_['n'],ev=s_['ev'],auc=s_['auc'],lo=s_['lo'],hi=s_['hi'],slope=s_['slope'],oe=s_['oe'],brier=s_['brier'])
alts={'primary (2010/4, 2014/4, 2019/4 blocks)':PRIMARY,'3-year blocks from 2012/4':['2012-04-01','2015-04-01','2018-04-01','2021-04-01','2024-04-01','2025-04-01'],'2-year blocks from 2013/4':['2013-04-01','2015-04-01','2017-04-01','2019-04-01','2021-04-01','2023-04-01','2025-04-01'],'single split at 2019/4':['2019-04-01','2025-04-01']}
sens['blocks']={}
for nm,bd in alts.items():
    sens['blocks'][nm]={m:srow(*run(kind,cols,bd)) for m,(kind,cols) in {'Ridge':('ridge',ALL_FEAT),'CCI':('lr0',['CCIa']),'ASA':('lr0',['ASA-PS'])}.items()}
sens['sets']={nm:srow(*run(k,c,PRIMARY)) for nm,(k,c) in {'Ridge, 12 predictors (primary)':('ridge',ALL_FEAT),'Ridge, without approach (open)':('ridge',[c for c in ALL_FEAT if c!='open']),'Ridge, 4 prespecified core predictors':('ridge',MANDATORY),'Unpenalized LR, 12 predictors':('lr0',ALL_FEAT)}.items()}
def run_slide(kind,cols,bounds,win=10):
    Pp=[];Yy=[]
    for i in range(len(bounds)-1):
        b0=pd.Timestamp(bounds[i]); tr=((d.date<b0)&(d.date>=b0-pd.DateOffset(years=win))).values; te=((d.date>=b0)&(d.date<pd.Timestamp(bounds[i+1]))).values
        Pp.append(fit_predict(kind,cols,np.where(tr)[0],np.where(te)[0])); Yy.append(y[te])
    return np.concatenate(Pp),np.concatenate(Yy)
p_,y_=run_slide('ridge',ALL_FEAT,PRIMARY,10); sens['sliding10']=dict(auc=roc_auc_score(y_,p_),ci=boot_ci(y_,p_,roc_auc_score,1000))
d2=d.copy(); inter=[]
for c in ALL_FEAT:
    if c!='open': d2['x_'+c]=d2[c]*d2['open']; inter.append('x_'+c)
dbl.d=d2; Pi=[];Yi=[]
for tr,te in blocks_from(PRIMARY): Pi.append(dbl.fit_predict('ridge',ALL_FEAT+inter,np.where(tr)[0],np.where(te)[0])); Yi.append(y[te])
Pi=np.concatenate(Pi);Yi=np.concatenate(Yi); dbl.d=d; sens['interaction']=dict(auc=roc_auc_score(Yi,Pi),ci=boot_ci(Yi,Pi,roc_auc_score,1000),slope=cal_metrics(Yi,Pi)[0])
sens['cvseed']=[roc_auc_score(*(lambda r:(r[1],r[0]))(run('ridge',ALL_FEAT,PRIMARY,s))) for s in range(5)]
rec=[]
for k in (1,2):
    tr_=bb<k; te_=bb==k; f=sm.Logit(yy[tr_],sm.add_constant(L[tr_])).fit(disp=0); pn=f.predict(sm.add_constant(L[te_],has_constant='add'))
    rec.append(dict(block=k+1,slope_before=cal_metrics(yy[te_],preds['Ridge'][te_])[0],slope_after=cal_metrics(yy[te_],pn)[0],brier_before=brier_score_loss(yy[te_],preds['Ridge'][te_]),brier_after=brier_score_loss(yy[te_],pn)))
sens['recal']=rec; F['sens']=sens
# ---------- final model on all patients: coefficients + bootstrap optimism ----------
cols=ALL_FEAT; pp=Prep().fit(d[cols]); X=pp.transform(d[cols]); Ximp=pp.imp.transform(d[cols])
cv=lambda: LogisticRegressionCV(Cs=np.logspace(-3,1,15),cv=StratifiedKFold(5,shuffle=True,random_state=SEED),scoring='neg_log_loss',max_iter=2000)
m=cv().fit(X,y); coef=m.coef_[0]; F['final_C']=float(m.C_[0]); F['intercept_std']=float(m.intercept_[0]); F['coef_std']=dict(zip(cols,coef.tolist()))
F['coef_raw']={c:float(coef[i]/pp.sc.scale_[i]) for i,c in enumerate(cols)}; F['intercept_raw']=float(m.intercept_[0]-sum(coef[i]*pp.sc.mean_[i]/pp.sc.scale_[i] for i in range(len(cols))))
F['scaler_mean']=dict(zip(cols,pp.sc.mean_.tolist())); F['scaler_sd']=dict(zip(cols,pp.sc.scale_.tolist()))
pa=m.predict_proba(X)[:,1]; app=roc_auc_score(y,pa); app_slope=cal_metrics(y,pa)[0]
B=300; cb=[];oa=[];osl=[]; r=np.random.default_rng(SEED)
for b in range(B):
    bi=r.choice(len(d),len(d))
    if len(np.unique(y[bi]))<2: continue
    ppb=Prep().fit(d.iloc[bi][cols]); Xb=ppb.transform(d.iloc[bi][cols]); Xo=ppb.transform(d[cols])
    mb=LogisticRegressionCV(Cs=np.logspace(-3,1,15),cv=5,scoring='neg_log_loss',max_iter=2000).fit(Xb,y[bi])
    # standardized coefficients on a common scale (original-scale coef x SD of the full sample)
    cb.append(mb.coef_[0]/ppb.sc.scale_*pp.sc.scale_); pb=mb.predict_proba(Xb)[:,1]; po=mb.predict_proba(Xo)[:,1]
    oa.append(roc_auc_score(y[bi],pb)-roc_auc_score(y,po)); osl.append(cal_metrics(y[bi],pb)[0]-cal_metrics(y,po)[0])
cb=np.array(cb); F['coef_ci']={c:np.percentile(cb[:,i],[2.5,97.5]).tolist() for i,c in enumerate(cols)}; F['coef_boot_B']=len(cb)
F['boot_val']=dict(apparent_auc=app,optimism=float(np.mean(oa)),corrected_auc=app-float(np.mean(oa)),apparent_slope=app_slope,corrected_slope=app_slope-float(np.mean(osl)))
# CCI-only apparent
pc=Prep().fit(d[['CCIa']]); lr=LogisticRegression().fit(pc.transform(d[['CCIa']]),y); F['cci_apparent_auc']=roc_auc_score(y,lr.predict_proba(pc.transform(d[['CCIa']]))[:,1])
# ---------- Riley ----------
def riley(prev,n_p,auc=0.65,sh=0.9):
    cs=2*(auc-0.5)*prev*(1-prev); n1=int(np.ceil(10*n_p/prev)); n2=int(np.ceil(n_p/(cs*(1-sh)))); mr=1-prev**prev*(1-prev)**(1-prev); n3=int(np.ceil(n_p/(0.05*cs/mr))); return dict(n1=n1,n2=n2,n3=n3,n=max(n1,n2,n3))
F['riley']=riley(y.mean(),12); F['prev_all']=float(y.mean())
# ---------- Table 1 by period ----------
per=[('2000-04-01','2010-04-01'),('2010-04-01','2014-04-01'),('2014-04-01','2019-04-01'),('2019-04-01','2025-04-01')]
grp=np.zeros(len(d),int)
for k,(a,b) in enumerate(per): grp[((d.date>=a)&(d.date<b)).values]=k
t1={'n':[int((grp==k).sum()) for k in range(4)],'n_all':len(d)}
def cont(c):
    out=[]
    for k in range(5):
        s=d[c][(grp==k)] if k<4 else d[c]; s=s.dropna(); out.append((float(s.median()),float(s.quantile(.25)),float(s.quantile(.75)),int(s.shape[0])))
    p=kruskal(*[d[c][grp==k].dropna() for k in range(4)])[1]; return out,float(p)
def cat(c,val=1):
    out=[]
    for k in range(5):
        s=d[c][(grp==k)] if k<4 else d[c]; s=s.dropna(); n=int((s==val).sum()); out.append((n,float(n/len(s)*100),int(len(s))))
    ct=pd.crosstab(d[c].dropna()[grp[d[c].notna().values]<4]==val if False else d.loc[d[c].notna(),c]==val,grp[d[c].notna().values]); return out,float(chi2_contingency(ct)[1])
for c in ['age','BMI','CCIa','CRP/Alb_ratio','PNI','PT-INR']: t1[c]=cont(c)
for c in ['sex','TG','open','history_of_laparotomy','simultaneous_operation','G2','G3']: t1[c]=cat(c)
asa=[]
for k in range(5):
    s=d['ASA-PS'][(grp==k)] if k<4 else d['ASA-PS']; vc=s.value_counts().sort_index(); asa.append(dict(counts={int(a):int(b) for a,b in vc.items()},missing=int(s.isna().sum())))
t1['ASA']=asa; ct=pd.crosstab(d['ASA-PS'].dropna().clip(1,4).replace({1:2}),grp[d['ASA-PS'].notna().values]); t1['ASA_p']=float(chi2_contingency(ct)[1]); F['t1']=t1
miss={}
for c in ['PT-INR','CRP/Alb_ratio','PNI','BMI','ASA-PS']:
    mm=d[c].isna(); miss[c]=dict(all=float(mm.mean()*100),n=int(mm.sum()),pre2010=float(mm[d.date<'2010-01-01'].mean()*100),post2010=float(mm[d.date>='2010-01-01'].mean()*100))
F['missing']=miss; F['n_total']=len(d); F['ev_total']=int(y.sum()); F['g3_total']=int(d.G3.sum())
pickle.dump(F,open('final_numbers.pkl','wb')); pickle.dump({'preds':preds,'yy':yy,'bb':bb,'open':open_,'dates':dates,'idx':idx},open('final_preds.pkl','wb'))
print('done'); print({k:(round(v['auc'],3),round(v['slope'],2)) for k,v in perf.items()}); print(F['boot_val'],F['riley'],F['holm'])
