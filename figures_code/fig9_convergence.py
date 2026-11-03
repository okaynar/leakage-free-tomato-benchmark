# Run from the repository root:  python figures_code/fig9_convergence.py
# Set TOMATO_ROOT to the dataset folder where raw images are needed.
import os
os.makedirs('figures', exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import os
import json
from scipy import stats


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


def f(x, d=3):
    return f"{x:.{d}f}"


# Load data
cv_raw = pd.read_csv('results/results_20models_cv_test.csv')

# Load cv_runs from handoff (need val_hist column)
# We reconstruct from the artifact which has cv results
# The artifact is results_20models_cv_test.csv which has per-model aggregates
# We need the raw cv_runs with val_hist - load from handoff if available
import io

# Try to load cv_runs with val_hist
try:
    raw_txt = open('results/cv_runs.csv', encoding='utf-8').read().replace('\\[', '[').replace('\\]', ']')
    cv = pd.read_csv(io.StringIO(raw_txt))
    cv.columns = [c.strip() for c in cv.columns]
    cv['model'] = cv.model.str.strip()
    cv['H'] = cv.val_hist.apply(ast.literal_eval)
except Exception:
    cv = None

T2 = cv_raw.copy()
if 'model' in T2.columns:
    T2 = T2.set_index('model')

# Group colors and mappings
GM = {'A: klasik TL': 'C-CNN', 'B: verimli CNN': 'M-CNN', 'C: ViT': 'ViT', 'D: hibrit': 'Hybrid'}
GC = {'C-CNN': '#8C8C8C', 'M-CNN': '#4C72B0', 'ViT': '#DD8452', 'Hybrid': '#55A868'}
NM = {'ConvNeXtV2-Tiny': 'ConvNeXt V2-T', 'InceptionNeXt-Tiny': 'InceptionNeXt-T',
      'GhostNetV2-1.0': 'GhostNetV2', 'MobileNetV4-Conv-M': 'MobileNetV4-M',
      'MobileViTv2-1.0': 'MobileViTv2'}
nm = lambda x: NM.get(x, x)

SEL = 'PVTv2-B2'

# Map group names
if 'group' in T2.columns:
    T2['grp'] = T2['group'].map(GM)
elif 'grp' not in T2.columns:
    T2['grp'] = 'C-CNN'

# Ensure numeric columns
for c in ['cv_f1', 'test_f1', 'ens_macro_f1', 'params_M', 'GFLOPs', 'latency_ms', 'ep']:
    if c in T2.columns:
        T2[c] = pd.to_numeric(T2[c], errors='coerce')

# MBConv models
MB = ['MobileNetV2', 'EfficientNet-B0', 'EfficientNetV2-S', 'MobileNetV4-Conv-M']

# Compute ensemble gain
if 'ens_gain' not in T2.columns and 'ens_macro_f1' in T2.columns and 'test_f1' in T2.columns:
    T2['ens_gain'] = T2['ens_macro_f1'] - T2['test_f1']

# Compute ep from cv if available
if cv is not None:
    ep_med = cv.groupby('model').best_epoch.median()
    T2['ep'] = ep_med.reindex(T2.index)

# Spearman between ep and ens_gain
rr = stats.spearmanr(T2['ep'].dropna(), T2.loc[T2['ep'].notna(), 'ens_gain'])

MAXE = 40


def curve(h):
    a_ = np.maximum.accumulate(np.array(h, float))
    out = np.full(MAXE, a_[-1])
    out[:len(a_)] = a_
    return out


plt.rcdefaults()
apply_figure_style(sizes=(10, 9, 9))
_tnr()

gorder = ['ViT', 'Hybrid', 'M-CNN', 'C-CNN']


fig = plt.figure(figsize=(6.7, 3.25))
sf_a, sf_b = fig.subfigures(1, 2, width_ratios=[1.2, 1], wspace=0.06)
gs = sf_a.add_gridspec(2, 2, wspace=0.12, hspace=0.22, left=0.13, right=0.98, bottom=0.15, top=0.86)
axs = [sf_a.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
for ax_, g in zip(axs, gorder):
    ms = T2[T2.grp == g].sort_values('cv_f1', ascending=False).index
    for j_, m_ in enumerate(ms):
        C = np.stack(cv[cv.model == m_].H.apply(curve).values).mean(0)
        ax_.plot(np.arange(1, MAXE + 1), C, color=GC[g], alpha=[1, .8, .62, .48, .35][j_],
                 lw=1.1, ls='--' if m_ in MB else '-')
    ax_.text(0.96, 0.08, g, transform=ax_.transAxes, ha='right', fontsize=9, color=GC[g], fontweight='bold')
    ax_.set_ylim(0.25, 0.9); ax_.set_xlim(1, MAXE)
    ax_.set_xticks([1, 10, 20, 30, 40]); ax_.set_yticks([0.4, 0.6, 0.8])
for ax_ in axs[2:]: ax_.set_xlabel('Epoch')
for ax_ in axs[:2]: ax_.set_xticklabels([])
for ax_ in [axs[1], axs[3]]: ax_.set_yticklabels([])
sf_a.supylabel('Best validation macro-F1', fontsize=10, x=0.02)
axs[0].plot([], [], color='0.35', ls='--', lw=1.1, label='MBConv-based')
sf_a.legend(*axs[0].get_legend_handles_labels(), loc='upper right', bbox_to_anchor=(0.98, 0.995), frameon=False, fontsize=9)
sf_a.text(0.02, 0.965, 'a', fontweight='bold', fontsize=11, va='top')

b = sf_b.add_axes([0.24, 0.15, 0.73, 0.71])
for m_, r_ in T2.iterrows():
    b.scatter(r_.ep, r_.ens_gain, s=22, color=GC[r_.grp], marker='D' if m_ in MB else 'o',
              edgecolor='black' if m_ == SEL else 'none', lw=0.8, zorder=3)
for m_, (ox, oy, ha) in {'MobileNetV2': (-6, 0, 'right'), 'MobileNetV4-Conv-M': (-6, 0, 'right'),
                          'EfficientNetV2-S': (5, -9, 'left'), 'EfficientNet-B0': (-6, 0, 'right'),
                          'ResNet50': (0, 8, 'center'),
                          'GhostNetV2-1.0': (0, -11, 'center')}.items():
    r_ = T2.loc[m_]
    b.annotate(nm(m_), (r_.ep, r_.ens_gain), xytext=(ox, oy), textcoords='offset points',
               fontsize=9, ha=ha, va='center')
r_ = T2.loc['PVTv2-B2']
b.annotate('PVTv2-B2', (r_.ep, r_.ens_gain), xytext=(8.0, 0.072), textcoords='data', fontsize=9, ha='left', va='center',
           arrowprops=dict(arrowstyle='-', lw=0.5, color='0.4', shrinkB=3))
b.axhline(0, color='0.6', lw=0.6)
b.set_xlabel('Median best epoch (CV)'); b.set_ylabel('Ensemble gain (macro-F1)')
b.set_xlim(7, 25); b.set_ylim(-0.03, 0.17); b.set_yticks([0, 0.05, 0.10, 0.15])
b.legend(handles=[mpl.lines.Line2D([], [], marker='D', ls='', color='0.4', ms=4.5, label='MBConv-based'),
                  mpl.lines.Line2D([], [], marker='o', ls='', color='0.4', ms=4.5, label='other')],
         frameon=False, loc='center right', fontsize=9, bbox_to_anchor=(1.0, 0.58))
b.text(0.98, 0.43, f'Spearman \u03c1 = {f(rr.statistic, 2)}\n($p$ = {f(rr.pvalue, 2)}, n = 20)',
       transform=b.transAxes, fontsize=9, va='top', ha='right')
sf_b.text(0.02, 0.965, 'b', fontweight='bold', fontsize=11, va='top')
fig.savefig('figures/Fig9_convergence.png', dpi=600)
