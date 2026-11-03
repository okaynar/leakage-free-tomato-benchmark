# Run from the repository root:  python figures_code/figS4_cv_protocol.py
# Set TOMATO_ROOT to the dataset folder where raw images are needed.
import os
os.makedirs('figures', exist_ok=True)
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, Patch
import pandas as pd
import numpy as np

def _tnr():
    import matplotlib as mpl, matplotlib.font_manager as fm
    for f in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
        p = 'C:/Windows/Fonts/' + f
        if os.path.exists(p): fm.fontManager.addfont(p)
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'], 'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
        'axes.unicode_minus': False})

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


apply_figure_style(sizes=(10, 9, 9))
_tnr()

m = pd.read_csv('splits/raw_clean_group_split.csv')
fs = m[m.split == 'trainval'].fold.value_counts().sort_index().tolist()
nte = (m.split == 'test').sum()

fig, ax = plt.subplots(figsize=(6.7, 3.10))
cTr, cVa, cTe, cUn = '#9DB4D6', '#DD8452', '#D9D9D9', '#FFFFFF'
rows = [f'Fold {i+1}' for i in range(5)] + ['Final training']
for r, lab in enumerate(rows):
    y = len(rows) - 1 - r
    x = 0
    for k, n in enumerate(fs):
        if lab == 'Final training':
            c = cTr
        else:
            c = cVa if k == r else cTr
        ax.barh(y, n, left=x, color=c, edgecolor='white', height=0.7)
        x += n
    x += 12
    ax.barh(y, nte, left=x, color=cTe if lab != 'Final training' else '#4D4D4D', edgecolor='white', height=0.7, hatch='////' if lab != 'Final training' else None)
    ax.text(-8, y, lab, ha='right', va='center', fontsize=9)
xt = sum(fs) + 12
ax.text(sum(fs) / 2, len(rows) - 0.35, f'Training/validation: {sum(fs)} images (5 group-disjoint folds: {"/".join(map(str, fs))})', ha='center', va='bottom', fontsize=9)
ax.text(xt + nte / 2, len(rows) - 0.35, f'Test: {nte}', ha='center', va='bottom', fontsize=9)
ax.text(xt + nte / 2, 0, 'single evaluation', ha='center', va='center', fontsize=9, color='white')
ax.plot([-4, xt + nte + 4], [0.5, 0.5], color='#888888', lw=0.6, ls='--')
ax.legend(handles=[Patch(fc=cTr, label='Training'), Patch(fc=cVa, label='Validation (early stopping, selection)'), Patch(fc=cTe, hatch='////', label='Test (locked)')],
          frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3)
ax.set_xlim(-60, xt + nte + 5)
ax.set_ylim(-0.6, len(rows) + 0.2)
ax.axis('off')
ax.text(xt + nte + 3, 3.5, '× 5 seeds\n(0–4)', ha='left', va='center', fontsize=9)
ax.set_xlim(-60, xt + nte + 55)
fig.savefig('figures/FigS4_cv_protocol.png', dpi=600, bbox_inches='tight')
print(fs, nte)