import pickle, numpy as np, statsmodels.api as sm
from scipy.stats import chi2
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import brier_score_loss
from statsmodels.stats.multitest import multipletests
F=pickle.load(open('final_numbers.pkl','rb')); Pd=pickle.load(open('final_preds.pkl','rb'))
p=np.clip(Pd['preds']['Ridge'],1e-6,1-1e-6); y=Pd['yy'].astype(int); op=Pd['open'].astype(int)
L=np.log(p/(1-p)); out={}
def fit(Lx,yx):
    return sm.Logit(yx,sm.add_constant(Lx)).fit(disp=0)
for name,m in (('open',op==1),('mis',op==0)):
    Ls,ys=L[m],y[m]; f=fit(Ls,ys); ci=f.conf_int()
    # calibration-in-the-large given slope 1
    f0=sm.GLM(ys,np.ones(len(ys)),family=sm.families.Binomial(),offset=Ls).fit(); cil=f0.params[0]; cil_ci=f0.conf_int()[0]
    rs=RepeatedStratifiedKFold(n_splits=5,n_repeats=50,random_state=1)
    bb,ba,bi=[],[],[]
    for tr,te in rs.split(Ls,ys):
        g=fit(Ls[tr],ys[tr]); pa=g.predict(sm.add_constant(Ls[te],has_constant='add'))
        g0=sm.GLM(ys[tr],np.ones(len(tr)),family=sm.families.Binomial(),offset=Ls[tr]).fit(); pi=1/(1+np.exp(-(Ls[te]+g0.params[0])))
        bb.append(brier_score_loss(ys[te],p[m][te])); ba.append(brier_score_loss(ys[te],pa)); bi.append(brier_score_loss(ys[te],pi))
    out[name]=dict(n=int(m.sum()),ev=int(ys.sum()),oe=float(p[m].sum()/ys.sum()) if False else float(ys.sum()/p[m].sum()),
        slope=float(f.params[1]),slope_ci=ci[1].tolist(),inter=float(f.params[0]),inter_ci=ci[0].tolist(),cil=float(cil),cil_ci=cil_ci.tolist(),
        brier_before=float(np.mean(bb)),brier_after_slope_int=float(np.mean(ba)),brier_after_int_only=float(np.mean(bi)),
        brier_obs_apparent=float(brier_score_loss(ys,p[m])))
# joint test of approach differences in calibration (intercept and slope)
X0=sm.add_constant(L); X1=sm.add_constant(np.column_stack([L,op,L*op]))
f0=sm.Logit(y,X0).fit(disp=0); f1=sm.Logit(y,X1).fit(disp=0)
lr=2*(f1.llf-f0.llf); out['joint_p']=float(chi2.sf(lr,2)); out['slope_int_p']=float(f1.pvalues[3])
fm=sm.Logit(y,sm.add_constant(np.column_stack([L,op]))).fit(disp=0); out['intercept_shift']=float(fm.params[2]); out['intercept_shift_p']=float(fm.pvalues[2])
# Holm over the two DeLong tests (primary incremental comparisons)
pl=[F['comp']['Ridge_vs_CCI']['p_delong'],F['comp']['Ridge_vs_ASA']['p_delong']]
out['holm2']=dict(zip(['CCI','ASA'],multipletests(pl,method='holm')[1].tolist()))
pickle.dump(out,open('approach_recal.pkl','wb'))
import json; print(json.dumps(out,indent=1,default=float))
