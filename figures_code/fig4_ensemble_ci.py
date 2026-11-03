# Run from the repository root:  python figures_code/fig4_ensemble_ci.py
# Set TOMATO_ROOT to the dataset folder where raw images are needed.
import os
os.makedirs('figures', exist_ok=True)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import math
import os
from scipy import stats
from statsmodels.stats.multitest import multipletests

def _tnr():
    import matplotlib as mpl, matplotlib.font_manager as fm
    for f in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
        p = 'C:/Windows/Fonts/' + f
        if os.path.exists(p): fm.fontManager.addfont(p)
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'], 'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
        'axes.unicode_minus': False})

# Global state
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

# Load data
T2 = pd.read_csv('results/results_20models_cv_test.csv').set_index('model')

# Load CV runs to compute piv and rk
cv = pd.read_csv('results/cv_runs.csv')
cv.columns = [c.strip() for c in cv.columns]
cv['model'] = cv.model.str.strip()

piv = cv.pivot_table(index=['seed','fold'], columns='model', values='macro_f1')
k, N = piv.shape[1], piv.shape[0]
rk = piv.rank(axis=1, ascending=False).mean()
CD = stats.studentized_range.ppf(0.95, k, np.inf) / math.sqrt(2) * math.sqrt(k*(k+1)/(6*N))

SEL = 'PVTv2-B2'

GC = {'C-CNN': '#8C8C8C', 'M-CNN': '#4C72B0', 'ViT': '#DD8452', 'Hybrid': '#55A868'}
GM = {'A: klasik TL': 'C-CNN', 'B: verimli CNN': 'M-CNN', 'C: ViT': 'ViT', 'D: hibrit': 'Hybrid'}

def nm(x):
    NM = {'ConvNeXtV2-Tiny': 'ConvNeXt V2-T', 'InceptionNeXt-Tiny': 'InceptionNeXt-T',
          'GhostNetV2-1.0': 'GhostNetV2', 'MobileNetV4-Conv-M': 'MobileNetV4-M',
          'MobileViTv2-1.0': 'MobileViTv2'}
    return NM.get(x, x)

# Ensure numeric types
for c in ['cv_f1', 'test_f1', 'ens_macro_f1', 'ens_mcc', 'params_M', 'GFLOPs', 'latency_ms', 'ep']:
    if c in T2.columns:
        T2[c] = pd.to_numeric(T2[c], errors='coerce')

T2['mcnemar_p_vs_selected'] = pd.to_numeric(T2['mcnemar_p_vs_selected'], errors='coerce')
T2['grp'] = T2['group'].map(GM)

# Compute Holm correction
fin_mc = T2.drop(SEL).mcnemar_p_vs_selected
rej, padj, _, _ = multipletests(fin_mc.values, method='holm')
T2.loc[fin_mc.index, 'mc_holm'] = padj

# Parse CI
ci = T2.ens_f1_ci95.astype(str).str.split('–', expand=True).astype(float)
T2['ci_lo'], T2['ci_hi'] = ci[0], ci[1]

nsig = int((T2.drop(SEL).mc_holm < 0.05).sum())

plt.rcdefaults()
apply_figure_style(sizes=(10, 9, 9))
_tnr()

E = T2.sort_values('ens_macro_f1', ascending=False).copy()
fig, ax = plt.subplots(figsize=(6.7, 3.0))
xs = np.arange(len(E))
for x_, (mname, r_) in zip(xs, E.iterrows()):
    c = GC[r_.grp]
    sig = (mname != SEL) and (r_.mc_holm < 0.05)
    ax.vlines(x_, r_.ci_lo, r_.ci_hi, color=c, lw=1.5, alpha=0.85)
    ax.scatter(x_, r_.ens_macro_f1, s=30 if mname == SEL else 22, marker='o',
               facecolor='white' if sig else c, edgecolor='black' if mname == SEL else c,
               lw=1.5 if mname == SEL else 1.1, zorder=3)
ax.axhline(E.loc[SEL, 'ens_macro_f1'], color='0.5', lw=0.7, ls='--', zorder=1)
ax.set_xticks(xs, [nm(m_) for m_ in E.index], rotation=45, ha='right', rotation_mode='anchor')
for t_, m_ in zip(ax.get_xticklabels(), E.index):
    if m_ == SEL: t_.set_fontweight('bold')
ax.set_xlim(-0.6, len(E) - 0.4)
ax.set_ylabel('Ensemble macro-F1 (95% CI)')
ax.set_ylim(0.6, 0.97); ax.set_yticks([0.6, 0.7, 0.8, 0.9])
ax.set_title(f'{len(E)-1-nsig} models are indistinguishable from the selected model (Holm-corrected McNemar)', loc='left', fontsize=10)
for g, c in GC.items(): ax.scatter([], [], color=c, s=18, label=g)
ax.scatter([], [], facecolor='white', edgecolor='0.3', s=18, label='Holm $p$ < 0.05')
ax.legend(frameon=False, loc='lower left', fontsize=9, ncol=3, bbox_to_anchor=(0.0, 0.0), columnspacing=0.8, handletextpad=0.2)
fig.savefig('figures/Fig4_ensemble_ci.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
