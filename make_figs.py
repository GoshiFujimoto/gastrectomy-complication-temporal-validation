import pickle, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
F=pickle.load(open('final_numbers.pkl','rb')); D=pickle.load(open('final_preds.pkl','rb'))
plt.rcParams.update({'font.family':'serif','font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.8,'figure.dpi':100})
COL={'Ridge':'#1f4e9c','CatBoost':'#e08214','CCI':'#4d4d4d','ASA':'#8073ac'}
LAB={'Ridge':'Ridge logistic regression','CatBoost':'CatBoost','CCI':'CCI only','ASA':'ASA-PS only'}
BL=['2010/4–2014/3','2014/4–2019/3','2019/4–2025/3']; BC=['#1b9e77','#d95f02','#7570b3']
import os; out=os.environ.get('OUT_DIR','figures/')+'' ; os.makedirs(out,exist_ok=True)
def save(fig,name):
    fig.savefig(f'/tmp/{name}.png',dpi=300,facecolor='white'); im=Image.open(f'/tmp/{name}.png').convert('RGB'); im.save(out+name+'.tif',dpi=(300,300),compression='tiff_lzw'); fig.savefig(out+name+'.pdf',facecolor='white'); plt.close(fig)
def panel(ax,t): ax.text(-0.14,1.06,t,transform=ax.transAxes,fontsize=12,fontweight='bold')
P=F['perf']
# Fig 1 ROC
fig,ax=plt.subplots(1,2,figsize=(7.2,3.6))
for nm in ['Ridge','CatBoost','CCI','ASA']:
    f,t,_=F['roc'][nm]; ax[0].step(f,t,where='post',color=COL[nm],lw=2.2 if nm=='Ridge' else 1.2,label=f"{LAB[nm]} (AUC = {P[nm]['auc']:.3f})")
ax[0].plot([0,1],[0,1],':',color='k',lw=0.8); ax[0].set_title(f"Pooled temporal-block validation\n(n = {F['n_test']}, events = {F['ev_test']})",fontsize=9)
for k in range(3):
    f,t,_=F['roc_block'][k]; ax[1].step(f,t,where='post',color=BC[k],lw=1.6,label=f"{BL[k]} (n = {F['block_n'][k]}; AUC = {P['Ridge']['block_auc'][k]:.3f})")
ax[1].plot([0,1],[0,1],':',color='k',lw=0.8); ax[1].set_title('Ridge logistic regression by validation block',fontsize=9)
for a,t in zip(ax,'ab'):
    a.set_xlabel('1 − Specificity'); a.set_ylabel('Sensitivity'); a.set_xlim(-0.02,1.02); a.set_ylim(-0.02,1.02); a.set_aspect('equal'); a.legend(loc='lower right',fontsize=6.8,frameon=True,framealpha=0.9,edgecolor='#cccccc'); panel(a,t)
fig.tight_layout(); save(fig,'Fig1_ROC_TemporalBlocks')
# Fig 2 calibration
fig,ax=plt.subplots(1,2,figsize=(7.2,3.6))
for nm,off in [('Ridge',0),('CatBoost',0.004)]:
    q=F['calq'][nm]; x=np.array([r['pred'] for r in q])+off; o=np.array([r['obs'] for r in q]); lo=np.array([r['lo'] for r in q]); hi=np.array([r['hi'] for r in q])
    ax[0].errorbar(x,o,yerr=[o-lo,hi-o],fmt='o-',color=COL[nm],lw=1.4,ms=4,capsize=2,label=f"{LAB[nm]} (slope {P[nm]['slope']:.2f})")
ax[0].plot([0,0.7],[0,0.7],'--',color='k',lw=0.8,label='Perfect calibration'); ax[0].set_xlim(0.1,0.6); ax[0].set_ylim(0,0.6); ax[0].set_title('Pooled validation (quintiles, 95% CI)',fontsize=9)
ax[0].set_xlabel('Mean predicted probability'); ax[0].set_ylabel('Observed proportion'); ax[0].legend(fontsize=7,loc='upper left',frameon=False); panel(ax[0],'a')
Pk=D['preds']['Ridge']; yy=D['yy']; bb=D['bb']; import pandas as pd
for k in range(3):
    m=bb==k; q=pd.qcut(Pk[m],3,labels=False); xs=[Pk[m][q==j].mean() for j in range(3)]; ys=[yy[m][q==j].mean() for j in range(3)]
    ax[1].plot(xs,ys,'o-',color=BC[k],lw=1.4,ms=4,label=f"{BL[k]} (slope {P['Ridge']['block_slope'][k]:.2f})")
ax[1].plot([0,0.7],[0,0.7],'--',color='k',lw=0.8); ax[1].set_xlim(0.1,0.6); ax[1].set_ylim(0,0.6); ax[1].set_title('Ridge logistic regression by block (tertiles)',fontsize=9)
ax[1].set_xlabel('Mean predicted probability'); ax[1].set_ylabel('Observed proportion'); ax[1].legend(fontsize=7,loc='upper left',frameon=False); panel(ax[1],'b')
fig.tight_layout(); save(fig,'Fig2_Calibration')
# Fig 3 DCA
th=np.array(F['dca']['th']); fig,ax=plt.subplots(1,2,figsize=(7.2,3.6))
for nm in ['Ridge','CatBoost','CCI','ASA']: ax[0].plot(th,F['dca'][nm],color=COL[nm],lw=2.2 if nm=='Ridge' else 1.2,label=LAB[nm])
ax[0].plot(th,F['dca']['all'],'--',color='k',lw=0.9,label='Treat all'); ax[0].axhline(0,color='k',ls=':',lw=0.9,label='Treat none'); ax[0].set_title('Pooled validation',fontsize=9)
ax[1].plot(th,F['dca_ridge_open'],color='#1f4e9c',lw=1.8,label=f"Ridge, open (n = {F['approach']['Ridge']['open'][0]})"); ax[1].plot(th,F['dca_ridge_mis'],color='#c0392b',lw=1.8,label=f"Ridge, minimally invasive (n = {F['approach']['Ridge']['mis'][0]})")
ax[1].plot(th,F['dca_all_open'],'--',color='#1f4e9c',lw=0.8,label='Treat all, open'); ax[1].plot(th,F['dca_all_mis'],'--',color='#c0392b',lw=0.8,label='Treat all, minimally invasive'); ax[1].axhline(0,color='k',ls=':',lw=0.9); ax[1].set_title('Ridge logistic regression by approach',fontsize=9)
for a,t in zip(ax,'ab'): a.set_xlim(0.05,0.6); a.set_ylim(-0.08,0.18); a.set_xlabel('Threshold probability'); a.set_ylabel('Net benefit'); a.legend(fontsize=6.6,frameon=False,loc='upper right'); panel(a,t)
fig.tight_layout(); save(fig,'Fig3_DCA')
# Fig 4 coefficients
lab={'age':'Age','sex':'Male sex','BMI':'Body mass index','ASA-PS':'ASA-PS','CCIa':'Age-adjusted CCI','history_of_laparotomy':'History of laparotomy','TG':'Total gastrectomy','open':'Open approach','simultaneous_operation':'Simultaneous organ resection','CRP/Alb_ratio':'CRP-to-albumin ratio','PNI':'PNI','PT-INR':'PT-INR'}
vars_=sorted(F['coef_std'],key=lambda c: abs(F['coef_std'][c]))
fig,ax=plt.subplots(figsize=(6.2,4.0))
for i,c in enumerate(vars_):
    b=F['coef_std'][c]; lo,hi=F['coef_ci'][c]; sd=F['scaler_sd'][c]
    # express as odds ratio per 1 SD using full-sample SD: boot CIs were computed on that scale
    ax.errorbar(np.exp(b),i,xerr=[[np.exp(b)-np.exp(lo)],[np.exp(hi)-np.exp(b)]],fmt='o',color='#1f4e9c',capsize=2.5,ms=4.5,lw=1.3)
ax.axvline(1,color='k',lw=0.8,ls='--'); ax.set_yticks(range(len(vars_))); ax.set_yticklabels([lab[c] for c in vars_]); ax.set_xscale('log'); ax.set_xticks([0.7,0.8,1,1.25,1.5,2]); ax.set_xticklabels(['0.7','0.8','1','1.25','1.5','2'])
ax.set_xlabel('Odds ratio per 1 SD (95% bootstrap CI)'); fig.tight_layout(); save(fig,'Fig4_Coefficients')
print('figs ok')
