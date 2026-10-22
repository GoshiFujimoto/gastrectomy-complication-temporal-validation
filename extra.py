import dbl, pickle, numpy as np
from dbl import *
from scipy.stats import chi2_contingency
out={}
d0=dbl.d.copy(); y0=dbl.y.copy()
for c in ['PT-INR','CRP/Alb_ratio']:
    m=d0[c].isna().astype(int); t=pd.crosstab(m,y0)
    out['miss_p_'+c]=chi2_contingency(t,correction=False)[1]
    out['miss_rate_by_G2_'+c]=[float(m[y0==0].mean()),float(m[y0==1].mean())]
cc=d0[ALL_FEAT].notna().all(axis=1).values
dc=d0[cc].reset_index(drop=True)
dbl.d=dc; dbl.y=dc.G2.astype(int).values
p,yy,bb=dbl.run('ridge',ALL_FEAT,PRIMARY)
s=dbl.summ(p,yy,bb); out['cc_ridge']=s; out['cc_n']=int(cc.sum()); out['cc_ev']=int(y0[cc].sum())
print({k:(v if k!='per' else v) for k,v in s.items()})
print(out['miss_p_PT-INR'],out['miss_p_CRP/Alb_ratio'],out['miss_rate_by_G2_PT-INR'],out['miss_rate_by_G2_CRP/Alb_ratio'],out['cc_n'])
pickle.dump(out,open('extra.pkl','wb'))
