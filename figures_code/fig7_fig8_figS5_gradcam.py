# Run from the repository root:  python figures_code/fig7_fig8_figS5_gradcam.py
# Set TOMATO_ROOT to the dataset folder where raw images are needed.
import os
os.makedirs('figures', exist_ok=True)

import os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use('Agg'); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from PIL import Image
for f in ['times.ttf','timesbd.ttf','timesi.ttf']: fm.fontManager.addfont('C:/Windows/Fonts/'+f)
mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif'],'font.size':9,'axes.titlesize':9,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'mathtext.fontset':'custom','mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic','axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
Z=np.load('results/gradcam_v7/gradcam_raw_7x7.npz'); PER=pd.read_csv('results/gradcam_per_image_stats.csv'); DEL=pd.read_csv('results/gradcam_v7/gradcam_deletion.csv')
ROOT=os.environ.get('TOMATO_ROOT', 'data/Tomato Dataset')
CL=['Early blight','Late blight','Septoria leaf spot','Leaf mold','TYLCV','Healthy']
def disp(rel):
    im=Image.open(os.path.normpath(os.path.join(ROOT,rel))).convert('RGB'); s=224/min(im.size)
    im=im.resize((max(224,round(im.width*s)),max(224,round(im.height*s))),Image.BICUBIC); l=(im.width-224)//2; t=(im.height-224)//2
    return np.asarray(im.crop((l,t,l+224,t+224)))
def up(c):
    c=np.asarray(Image.fromarray(c.astype(np.float32),mode='F').resize((224,224),Image.BILINEAR)); c=c-c.min(); return c/(c.max()+1e-8)
def ens(m,which,i): return np.mean([up(Z[f'{m}__{which}'][s,i]) for s in range(5)],0)
def ov(ax,i,cam=None):
    ax.imshow(disp(PER.relpath[i]))
    if cam is not None: ax.imshow(cam,cmap='jet',alpha=0.45,vmin=0,vmax=1)
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
pp,ps,y=PER['PVTv2-B2_pred'].values,PER['Swin-T_pred'].values,PER.y_true.values
rng=np.random.default_rng(0); pick=[]
for c in range(6):
    ok=np.where((y==c)&(pp==c)&(ps==c))[0]; pick.append(rng.choice(ok))
fig,axs=plt.subplots(3,6,figsize=(6.7,3.75))
for j,i in enumerate(pick):
    ov(axs[0,j],i); axs[0,j].set_title(CL[j],fontsize=9)
    ov(axs[1,j],i,ens('PVTv2-B2','pred',i)); ov(axs[2,j],i,ens('Swin-T','pred',i))
for r,lab in enumerate(['Image','PVTv2-B2','Swin-T']): axs[r,0].set_ylabel(lab,fontsize=10)
fig.subplots_adjust(wspace=0.04,hspace=0.06,left=0.06,right=0.99,top=0.93,bottom=0.01)
fig.savefig('figures/Fig7_gradcam_examples.png',dpi=600,bbox_inches='tight'); plt.close(fig)
# quantitative figure
fig,(a,b)=plt.subplots(1,2,figsize=(6.7,2.9),gridspec_kw={'width_ratios':[1.15,1]})
for m,col in zip(['PVTv2-B2','Swin-T'],['#DD8452','#4C72B0']):
    d=DEL[DEL.model==m]; base=d.base_prob.iloc[0]; x=[0]+list(d.frac)
    a.plot(x,[base]+list(base-d.drop_cam),'-o',color=col,ms=3.5,label=f'{m}, Grad-CAM')
    a.plot(x,[base]+list(base-d.drop_random),'--s',color=col,ms=3.5,alpha=0.75,label=f'{m}, random')
a.set_xlabel('Fraction of image area blurred'); a.set_ylabel('Mean probability of predicted class'); a.legend(frameon=False,loc='lower left'); a.set_ylim(0.3,0.82)
a.text(-0.16,1.02,'a',transform=a.transAxes,fontweight='bold',fontsize=10)
data=[PER['PVTv2-B2_seed_r'],PER['Swin-T_seed_r'],PER['cross_model_r']]
bp=b.boxplot(data,widths=0.5,showfliers=False,patch_artist=True,medianprops=dict(color='black'))
for p_,c in zip(bp['boxes'],['#DD8452','#4C72B0','#8C8C8C']): p_.set_facecolor(c); p_.set_alpha(0.55)
for k,v in enumerate(data): b.scatter(np.random.default_rng(k).normal(k+1,0.06,len(v)),v,s=4,color='0.25',alpha=0.5,lw=0)
b.set_xticks([1,2,3],['PVTv2-B2\nseed–seed','Swin-T\nseed–seed','PVTv2-B2\nvs Swin-T']); b.set_ylabel('Pearson r between CAMs'); b.set_ylim(-0.3,1.02)
b.text(-0.2,1.02,'b',transform=b.transAxes,fontweight='bold',fontsize=10)
fig.tight_layout(); fig.savefig('figures/Fig8_gradcam_quantitative.png',dpi=600,bbox_inches='tight'); plt.close(fig)
# errors (supplementary)
err=np.where(pp!=y)[0]; nc=2; nr=int(np.ceil(len(err)/nc))
fig,axs=plt.subplots(nr,nc*3,figsize=(6.7,1.18*nr+0.35))
for k,i in enumerate(err):
    r_,c_=divmod(k,nc); A=axs[r_,c_*3:c_*3+3]
    ov(A[0],i); ov(A[1],i,ens('PVTv2-B2','pred',i)); ov(A[2],i,ens('PVTv2-B2','true',i))
    A[0].text(0,1.05,f'{CL[y[i]]} → {CL[pp[i]]}',transform=A[0].transAxes,fontsize=9,ha='left',va='bottom')
for k in range(len(err),nr*nc):
    r_,c_=divmod(k,nc); [ax.axis('off') for ax in axs[r_,c_*3:c_*3+3]]
for c_ in range(nc):
    for t_,lab in enumerate(['Image','Predicted-class CAM','True-class CAM']): axs[-1,c_*3+t_].set_xlabel(lab,fontsize=9)
fig.subplots_adjust(wspace=0.04,hspace=0.28,left=0.01,right=0.99,top=0.96,bottom=0.05)
fig.savefig('figures/FigS5_gradcam_errors.png',dpi=600,bbox_inches='tight'); plt.close(fig)
print('picks',pick,'errors',len(err))
