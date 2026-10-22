from lib import *
from sklearn.linear_model import LogisticRegressionCV
import sys, time
n0,d=load_cohort(); d=d.reset_index(drop=True); y=d.G2.astype(int).values
try: SEL10=pickle.load(open('main_res.pkl','rb'))['sel']   # only needed for the 'sel10' variant (requires original_single_split/main.py output)
except FileNotFoundError: SEL10=ALL_FEAT
SETS={'all12':ALL_FEAT,'sel10':SEL10,'mand4':MANDATORY}
CATP=dict(iterations=200,depth=3,learning_rate=0.03,l2_leaf_reg=5,verbose=0,thread_count=2)
def fit_predict(kind,cols,tr,te,seed=SEED):
    pp=Prep().fit(d.loc[tr,cols]); Xa=pp.transform(d.loc[tr,cols]); Xb=pp.transform(d.loc[te,cols])
    if kind=='ridge':
        m=LogisticRegressionCV(Cs=np.logspace(-3,1,15),cv=StratifiedKFold(5,shuffle=True,random_state=seed),scoring='neg_log_loss',max_iter=2000).fit(Xa,y[tr])
    elif kind=='lr0': m=LogisticRegression(C=1e6,max_iter=2000).fit(Xa,y[tr])
    elif kind=='cat': m=CatBoostClassifier(random_seed=seed,**CATP).fit(Xa,y[tr])
    return m.predict_proba(Xb)[:,1]
def blocks_from(bounds):
    b=[pd.Timestamp(x) for x in bounds]; return [((d.date<b[i]).values,((d.date>=b[i])&(d.date<b[i+1])).values) for i in range(len(b)-1)]
def run(kind,cols,bounds,seed=SEED):
    P=[];Y=[];B=[]
    for k,(tr,te) in enumerate(blocks_from(bounds)):
        p=fit_predict(kind,cols,np.where(tr)[0],np.where(te)[0],seed); P.append(p);Y.append(y[te]);B.append(np.full(te.sum(),k))
    return np.concatenate(P),np.concatenate(Y),np.concatenate(B)
def summ(p,yy,bb):
    s,i,br,oe=cal_metrics(yy,p); lo,hi=boot_ci(yy,p,roc_auc_score,1000)
    per=[(int((bb==k).sum()),int(yy[bb==k].sum()),roc_auc_score(yy[bb==k],p[bb==k]),cal_metrics(yy[bb==k],p[bb==k])[0]) for k in np.unique(bb)]
    return dict(auc=roc_auc_score(yy,p),lo=lo,hi=hi,pr=average_precision_score(yy,p),brier=br,slope=s,int=i,oe=oe,per=per,n=len(yy),ev=int(yy.sum()))
PRIMARY=['2010-04-01','2014-04-01','2019-04-01','2025-04-01']
