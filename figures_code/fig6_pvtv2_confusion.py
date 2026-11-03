# Run from the repository root:  python figures_code/fig6_pvtv2_confusion.py
# Set TOMATO_ROOT to the dataset folder where raw images are needed.
import os
os.makedirs('figures', exist_ok=True)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def _tnr():
    import matplotlib as mpl, matplotlib.font_manager as fm
    for f in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
        p = 'C:/Windows/Fonts/' + f
        if os.path.exists(p): fm.fontManager.addfont(p)
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'], 'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
        'axes.unicode_minus': False})

# skill:figure-style kernel.py (auto-injected on skill load)
META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")

    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
    import matplotlib.pyplot as plt
    if fontsize is None:
        fontsize = plt.rcParams.get("font.size", 8) + 1
    s = letter.lower() if case == "lower" else letter.upper()
    ax.text(dx, dy, s, transform=ax.transAxes,
            fontweight="bold", fontsize=fontsize, va="bottom", ha="left")


apply_figure_style(sizes=(10, 9, 9))
_tnr()

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, precision_recall_fscore_support

CL=['Early blight','Late blight','Septoria leaf spot','Leaf mold','Yellow leaf curl virus','Healthy']
CM=np.array([[17,0,2,1,1,0],[1,19,0,0,0,0],[0,0,34,0,0,0],[1,0,2,33,1,0],[0,0,0,0,15,0],[1,0,0,0,2,20]])
y=np.repeat(np.arange(6),CM.sum(1)); p=np.concatenate([np.repeat(np.arange(6),r) for r in CM])
P,R,F,Nn=precision_recall_fscore_support(y,p)

TRs=['Early blight','Late blight','Septoria','Leaf mold','TYLCV','Healthy']
fig,(a,b)=plt.subplots(1,2,figsize=(6.7,3.53),gridspec_kw=dict(width_ratios=[1.15,1]))
rown=CM/CM.sum(1,keepdims=True)
a.imshow(rown,cmap='Blues',vmin=0,vmax=1)
for i in range(6):
    for j in range(6):
        if CM[i,j]: a.text(j,i,CM[i,j],ha='center',va='center',fontsize=9,color='white' if rown[i,j]>0.55 else 'black')
a.set_xticks(range(6),TRs,rotation=40,ha='right'); a.set_yticks(range(6),[f'{t} ({n})' for t,n in zip(TRs,CM.sum(1))])
a.set_xlabel('Predicted class'); a.set_ylabel('True class (n)'); a.set_title('Errors concentrate in early blight, leaf mold',loc='left')
yy=np.arange(6)[::-1]
for k_,(lab,off,mk) in enumerate([('Precision',0.22,'s'),('Recall',0,'o'),('F1',-0.22,'D')]):
    v=[P,R,F][k_]; b.scatter(v,yy+off,marker=mk,s=18,color=['0.55','0.25','#DD8452'][k_],label=lab,zorder=3)
b.hlines(yy-0.22,0.7,F,color='#DD8452',lw=0.6,alpha=0.5)
b.set_yticks(yy,TRs); b.set_xlim(0.7,1.02); b.set_xlabel('Value (150 test images, 5-seed ensemble)')
b.set_title('F1 ≥ 0.83 across all classes',loc='left'); b.legend(frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.2),ncol=3,fontsize=9); b.set_ylim(-0.6,5.6)
panel_letter(a,'a'); panel_letter(b,'b')
fig.tight_layout(); fig.savefig('figures/Fig6_pvtv2_confusion.png',dpi=600,bbox_inches='tight')