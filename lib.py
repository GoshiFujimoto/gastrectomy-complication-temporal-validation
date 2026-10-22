import warnings, re, json, pickle, os
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, scipy.stats as st
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.calibration import _SigmoidCalibration
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, confusion_matrix
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

SEED = 42
CUTOFF = pd.Timestamp('2019-04-01')
MANDATORY = ['ASA-PS','TG','history_of_laparotomy','CCIa']
OPTIONAL = ['age','sex','BMI','open','simultaneous_operation','CRP/Alb_ratio','PNI','PT-INR']
ALL_FEAT = MANDATORY + OPTIONAL
MODELS = ['LR','RF','XGB','LGBM','CAT','MLP']

def _exclude_ids(path=os.environ.get('GASTRECT_EXCLUDE_IDS', 'data/exclude_ids.txt')):
    # patient-level exclusions are kept in a local, untracked file (one ID per line); no identifiers are stored in the repository
    return tuple(int(x) for x in open(path).read().split()) if os.path.exists(path) else ()

def load_cohort(path=os.environ.get('GASTRECT_DATA', 'data/cohort.pkl'), drop_ids=None):
    if drop_ids is None: drop_ids = _exclude_ids()
    df = pd.read_pickle(path)
    d = df[(df['age']>=75)&(df['術日']>='2000-04-01')&(df['術日']<'2025-04-01')].copy()
    n0 = len(d)
    s = d['術式'].astype(str)
    pat = '試験開腹|staging|lap観察|bypass|局所切除|部分切除|胃部分|lesion|胃内手術|胃瘻|腸瘻'
    d = d[~s.str.contains(pat, case=False)].copy()
    if len(drop_ids): d = d[~d['ID'].isin(drop_ids)]
    d = d.drop(columns=['CCI']).rename(columns={'CCI(年齢調整)':'CCIa','Grade≧2':'G2','Grade≧3':'G3','術日':'date'})
    d['sex'] = (d['sex']=='M').astype(int)
    d['ASA-PS'] = pd.to_numeric(d['ASA-PS'].astype(str).str.extract(r'([1-4１-４])')[0].str.translate(str.maketrans('１２３４','1234')), errors='coerce')
    d['open'] = d['open'].astype(float)
    d = d.sort_values('date').reset_index(drop=True)
    return n0, d

class Prep:
    def __init__(self):
        self.imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=SEED)
        self.sc = StandardScaler()
    def fit(self, X):
        Xi = self.imp.fit_transform(X); self.sc.fit(Xi); return self
    def transform(self, X):
        return self.sc.transform(self.imp.transform(X))
    def fit_transform(self, X): return self.fit(X).transform(X)

def select_features(Xsc, y, Ximp_df, names, mandatory, optional, min_epv=10):
    oi = [names.index(c) for c in optional]
    l1 = LogisticRegression(C=0.1, penalty='l1', solver='liblinear', max_iter=1000, random_state=SEED).fit(Xsc[:, oi], y)
    l1c = pd.Series(np.abs(l1.coef_[0]), index=optional); l1s = set(l1c[l1c>0].index)
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=2).fit(Ximp_df[optional], y)
    rfi = pd.Series(rf.feature_importances_, index=optional); rfs = set(rfi[rfi>=rfi.median()].index)
    both = sorted(l1s & rfs, key=lambda v: rfi[v]+l1c[v]/max(l1c.max(),1e-9), reverse=True)
    l1o = sorted(l1s-rfs, key=lambda v: l1c[v], reverse=True)
    rfo = sorted(rfs-l1s, key=lambda v: rfi[v], reverse=True)
    ranked = both+l1o+rfo
    ev = int(y.sum()); n_add = min(4, len(ranked))
    for k in range(len(ranked), 3, -1):
        if ev/(len(mandatory)+k) >= min_epv: n_add = k; break
    return mandatory+ranked[:n_add]

def factories():
    return {
     'LR':  lambda: LogisticRegression(max_iter=1000, random_state=SEED),
     'RF':  lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=2),
     'XGB': lambda: XGBClassifier(eval_metric='logloss', random_state=SEED, n_jobs=2),
     'LGBM':lambda: LGBMClassifier(random_state=SEED, n_jobs=2, verbose=-1),
     'CAT': lambda: CatBoostClassifier(random_state=SEED, verbose=0, thread_count=2),
     'MLP': lambda: MLPClassifier(max_iter=500, random_state=SEED)}
lu = st.loguniform
PFAST = {
 'LR':{'C':lu(1e-3,10),'penalty':['l1','l2'],'solver':['liblinear']},
 'RF':{'n_estimators':[100,200],'max_depth':[None,3,5],'min_samples_leaf':[1,2],'max_features':['sqrt']},
 'XGB':{'n_estimators':[50,100],'max_depth':[3,5],'learning_rate':lu(0.01,0.3),'subsample':[0.8,1.0]},
 'LGBM':{'n_estimators':[50,100],'num_leaves':[15,31],'learning_rate':lu(0.01,0.3),'min_child_samples':[5,10]},
 'CAT':{'iterations':[50,100],'depth':[3,5],'learning_rate':lu(0.01,0.3)},
 'MLP':{'hidden_layer_sizes':[(32,),(64,)],'alpha':lu(1e-4,0.1),'learning_rate_init':lu(1e-4,1e-2)}}
PFULL = {
 'LR':PFAST['LR'],
 'RF':{'n_estimators':[100,200,300],'max_depth':[None,3,5,7],'min_samples_leaf':[1,2,5],'max_features':['sqrt','log2']},
 'XGB':{'n_estimators':[50,100,200],'max_depth':[3,5,7],'learning_rate':lu(0.01,0.3),'subsample':[0.6,0.8,1.0],'colsample_bytree':[0.6,0.8,1.0]},
 'LGBM':{'n_estimators':[50,100,200],'num_leaves':[15,31,63],'learning_rate':lu(0.01,0.3),'min_child_samples':[5,10,20]},
 'CAT':{'iterations':[50,100,200],'depth':[3,5,7],'learning_rate':lu(0.01,0.3)},
 'MLP':{'hidden_layer_sizes':[(32,),(64,),(64,32),(128,64)],'alpha':lu(1e-4,0.1),'learning_rate_init':lu(1e-4,1e-2)}}

def tune(name, X, y, grid, n_iter, seed=SEED):
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    rs = RandomizedSearchCV(factories()[name](), grid[name], n_iter=n_iter, cv=cv, scoring='roc_auc', random_state=seed, n_jobs=1, error_score=0)
    rs.fit(X, y); return rs.best_estimator_

def resample(X, y, seed=SEED):
    try: return SMOTEENN(random_state=seed).fit_resample(X, y)
    except Exception: return X, np.asarray(y)

class Platt:
    def __init__(self, base): self.base=base; self.s=_SigmoidCalibration()
    def fit(self, X, y):
        self.s.fit(self.base.predict_proba(X)[:,1].reshape(-1,1), y); return self
    def predict(self, X):
        return np.clip(self.s.predict(self.base.predict_proba(X)[:,1].reshape(-1,1)),1e-6,1-1e-6)

def cal_metrics(y, p):
    eps=1e-6; lg=np.log(np.clip(p,eps,1-eps)/(1-np.clip(p,eps,1-eps)))
    m=LogisticRegression(C=1e6, max_iter=1000).fit(lg.reshape(-1,1), y)
    return float(m.coef_[0][0]), float(m.intercept_[0]), brier_score_loss(y,p), float(np.mean(y)/np.mean(p))

def boot_ci(y, p, fn, n=1000, seed=SEED):
    rng=np.random.default_rng(seed); v=[]
    for _ in range(n):
        i=rng.integers(0,len(y),len(y))
        if len(np.unique(y[i]))<2: continue
        v.append(fn(y[i],p[i]))
    return float(np.percentile(v,2.5)), float(np.percentile(v,97.5))

# DeLong (Sun & Xu 2014)
def _compute_midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1); i=j
    T2=np.empty(N); T2[J]=T+1; return T2
def _fast_delong(preds, m):
    n=preds.shape[1]-m; k=preds.shape[0]
    pos=preds[:,:m]; neg=preds[:,m:]
    tx=np.array([_compute_midrank(p) for p in pos]); ty=np.array([_compute_midrank(p) for p in neg]); tz=np.array([_compute_midrank(p) for p in preds])
    aucs=tz[:,:m].sum(axis=1)/m/n-(m+1.0)/2.0/n
    v01=(tz[:,:m]-tx)/n; v10=1.0-(tz[:,m:]-ty)/m
    sx=np.cov(v01); sy=np.cov(v10)
    return aucs, sx/m+sy/n
def delong_test(y, p1, p2):
    y=np.asarray(y); order=np.argsort(-y, kind='stable'); m=int((y==1).sum())
    preds=np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    aucs,S=_fast_delong(preds,m); d=aucs[0]-aucs[1]; var=S[0,0]+S[1,1]-2*S[0,1]
    z=d/np.sqrt(var) if var>0 else 0.0
    return aucs[0],aucs[1],float(2*(1-st.norm.cdf(abs(z))))

def nri_idi(y, p_new, p_old):
    y=np.asarray(y); up=p_new>p_old; dn=p_new<p_old
    nri=(up[y==1].mean()-dn[y==1].mean())+(dn[y==0].mean()-up[y==0].mean())
    idi=(p_new[y==1].mean()-p_new[y==0].mean())-(p_old[y==1].mean()-p_old[y==0].mean())
    return nri, idi
