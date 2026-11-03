# Run from the repository root:  python figures_code/fig3_cv_results.py
# Set TOMATO_ROOT to the dataset folder where raw images are needed.
import os
os.makedirs('figures', exist_ok=True)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
import pandas as pd
import math
import os
import re
import glob
from scipy import stats
from PIL import Image


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


def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
    import matplotlib.pyplot as plt
    if fontsize is None:
        fontsize = plt.rcParams.get("font.size", 8) + 1
    s = letter.lower() if case == "lower" else letter.upper()
    ax.text(dx, dy, s, transform=ax.transAxes,
            fontweight="bold", fontsize=fontsize, va="bottom", ha="left")


# Load CV runs from Drive results
cv = pd.read_csv('results/cv_runs.csv')
cv.columns = [c.strip() for c in cv.columns]
for c in cv.select_dtypes('object'):
    cv[c] = cv[c].str.strip()

apply_figure_style(sizes=(10, 9, 9))
_tnr()

GM = {'A: klasik TL': 'C-CNN', 'B: verimli CNN': 'M-CNN', 'C: ViT': 'ViT', 'D: hibrit': 'Hybrid'}
GC = {'C-CNN': '#8C8C8C', 'M-CNN': '#4C72B0', 'ViT': '#DD8452', 'Hybrid': '#55A868'}
NM = {'ConvNeXtV2-Tiny': 'ConvNeXt V2-T', 'InceptionNeXt-Tiny': 'InceptionNeXt-T', 'GhostNetV2-1.0': 'GhostNetV2', 'MobileNetV4-Conv-M': 'MobileNetV4-M', 'MobileViTv2-1.0': 'MobileViTv2'}
nm = lambda x: NM.get(x, x)
cv['grp'] = cv.group.map(GM)
SEL = 'PVTv2-B2'
order = cv.groupby('model').macro_f1.mean().sort_values(ascending=False).index.tolist()
grp_of = cv.groupby('model').grp.first()
piv = cv.pivot_table(index=['seed', 'fold'], columns='model', values='macro_f1')
k, N = piv.shape[1], piv.shape[0]
rk = piv.rank(axis=1, ascending=False).mean().sort_values()
qa = stats.studentized_range.ppf(0.95, k, np.inf) / math.sqrt(2)
CD = qa * math.sqrt(k * (k + 1) / (6 * N))
fr = stats.friedmanchisquare(*[piv[c] for c in piv])
fig = plt.figure(figsize=(6.7, 4.1))
sfa, sfb = fig.subfigures(1, 2, width_ratios=[1.08, 1], wspace=0.05)
a = sfa.add_axes([0.35, 0.12, 0.62, 0.80]); b = sfb.add_axes([0.38, 0.12, 0.59, 0.80])
n = len(order); ya = np.arange(n)[::-1]
for y_, mname in zip(ya, order):
    v = cv[cv.model == mname].macro_f1.values
    c = GC[grp_of[mname]]
    a.scatter(v, np.random.default_rng(y_).normal(y_, 0.07, len(v)), s=4, color=c, alpha=0.55, lw=0, zorder=2)
    a.boxplot([v], positions=[y_], widths=0.6, orientation='horizontal', showfliers=False, patch_artist=True,
              boxprops=dict(facecolor='none', edgecolor='black' if mname == SEL else '0.35', lw=1.3 if mname == SEL else 0.7),
              medianprops=dict(color='black', lw=1.1), whiskerprops=dict(color='0.35', lw=0.7), capprops=dict(color='0.35', lw=0.7))
a.set_yticks(ya, [nm(x) for x in order]); a.set_ylim(-0.7, n - 0.3)
for tl, mname in zip(a.get_yticklabels(), order):
    if mname == SEL: tl.set_fontweight('bold')
a.set_xlabel('Validation macro-F1 (25 runs)'); a.set_xlim(0.52, 0.92); a.set_xticks([0.6, 0.7, 0.8, 0.9])
for g, c in GC.items(): a.scatter([], [], color=c, s=14, label=g)
a.legend(frameon=False, loc='upper left', fontsize=9, handletextpad=0.2, borderaxespad=0.2)
sfa.text(0.01, 0.995, 'a', fontweight='bold', fontsize=11, va='top')
sfa.text(0.07, 0.995, 'CV macro-F1 by model', fontsize=10, va='top')
ys = np.arange(len(rk))[::-1]
best = rk.min()
b.axvspan(best, best + CD, color='#DD8452', alpha=0.12, lw=0)
b.hlines(ys, rk.min() - 0.3, rk.values, color='0.85', lw=0.6, zorder=1)
for y_, (mname, r_) in zip(ys, rk.items()):
    b.scatter(r_, y_, s=18, color=GC[grp_of[mname]], edgecolor='black' if mname == SEL else 'none', lw=0.9, zorder=3)
b.set_yticks(ys, [nm(x) for x in rk.index])
for tl, mname in zip(b.get_yticklabels(), rk.index):
    if mname == SEL: tl.set_fontweight('bold')
b.annotate('', xy=(best + CD, len(rk) - 0.1), xytext=(best, len(rk) - 0.1), arrowprops=dict(arrowstyle='<->', lw=0.7, color='0.3'))
b.text(best + CD + 0.4, len(rk) - 0.1, f'CD = {CD:.2f}', ha='left', va='center', fontsize=9)
b.set_xlabel('Mean rank (lower = better)')
b.set_ylim(-0.7, len(rk) + 0.6); b.set_xlim(rk.min() - 0.5, rk.max() + 0.7); b.set_xticks([5, 10, 15])
sfb.text(0.01, 0.995, 'b', fontweight='bold', fontsize=11, va='top')
sfb.text(0.07, 0.995, f'Friedman rank; {int((rk - best <= CD).sum())} models within CD', fontsize=10, va='top')
fig.savefig('figures/Fig3_cv_results.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
print(round(fr.statistic, 1), fr.pvalue, round(CD, 2), round(qa, 3), k, N, int((rk - best <= CD).sum()))
