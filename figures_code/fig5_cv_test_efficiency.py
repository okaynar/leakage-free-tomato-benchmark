# Run from the repository root:  python figures_code/fig5_cv_test_efficiency.py
# Set TOMATO_ROOT to the dataset folder where raw images are needed.
import os
os.makedirs('figures', exist_ok=True)
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


apply_figure_style(sizes=(10, 9, 9))
_tnr()

# Load data
T = pd.read_csv('results/results_20models_cv_test.csv')
T = T.set_index('model')

# Load cv_runs from handoff (results/ folder of this repository)
cv = pd.read_csv('results/cv_runs.csv')
cv.columns = [c.strip() for c in cv.columns]
for c in cv.select_dtypes('object'):
    cv[c] = cv[c].str.strip()

fin = pd.read_csv('results/final_test_runs.csv')
fin.columns = [c.strip() for c in fin.columns]
for c in fin.select_dtypes('object'):
    fin[c] = fin[c].str.strip()

GM = {'A: klasik TL': 'C-CNN', 'B: verimli CNN': 'M-CNN', 'C: ViT': 'ViT', 'D: hibrit': 'Hybrid'}
GC = {'C-CNN': '#8C8C8C', 'M-CNN': '#4C72B0', 'ViT': '#DD8452', 'Hybrid': '#55A868'}
NM = {'ConvNeXtV2-Tiny': 'ConvNeXt V2-T', 'InceptionNeXt-Tiny': 'InceptionNeXt-T',
      'GhostNetV2-1.0': 'GhostNetV2', 'MobileNetV4-Conv-M': 'MobileNetV4-M',
      'MobileViTv2-1.0': 'MobileViTv2'}
nm = lambda x: NM.get(x, x)

cv['grp'] = cv.group.map(GM)
SEL = 'PVTv2-B2'

T2 = T.copy()
T2['grp'] = T2.group.map(GM)
fin_sd = fin.groupby('model').macro_f1.std()

W, Hf = 6.7, 3.45
fig = plt.figure(figsize=(W, Hf))
AH, BOT = 2.6, 0.5
a = fig.add_axes([0.6 / W, BOT / Hf, AH / W, AH / Hf])
b = fig.add_axes([3.9 / W, BOT / Hf, 2.65 / W, AH / Hf])
lo, hi = 0.52, 0.92
a.plot([lo, hi], [lo, hi], color='0.6', lw=0.8, ls='--', zorder=1)
a.text(0.565, 0.585, 'y = x', fontsize=9, color='0.4', rotation=38)
for mname, r_ in T2.iterrows():
    a.errorbar(r_.cv_f1, r_.test_f1, xerr=r_.cv_sd, yerr=fin_sd[mname], fmt='none', ecolor=GC[r_.grp], alpha=0.45, lw=0.7, zorder=2)
    a.scatter(r_.cv_f1, r_.test_f1, s=24, color=GC[r_.grp], edgecolor='black' if mname == SEL else 'none', lw=1, zorder=3)
rho = stats.spearmanr(T2.cv_f1, T2.test_f1)
for mname, dx, dy in [('PVTv2-B2', -0.075, 0.018), ('MobileNetV2', 0.015, -0.02), ('ResNet50', -0.09, 0.02), ('GhostNetV2-1.0', 0.012, -0.035), ('MobileViTv2-1.0', 0.005, -0.05)]:
    r_ = T2.loc[mname]
    a.annotate(nm(mname), (r_.cv_f1, r_.test_f1), xytext=(r_.cv_f1 + dx, r_.test_f1 + dy), fontsize=9, arrowprops=dict(arrowstyle='-', lw=0.5, color='0.4'))
a.text(0.535, 0.89, f'Spearman ρ = {rho.statistic:.3f}\nn = 20 models', fontsize=9, va='top')
a.set_xlim(lo, hi); a.set_ylim(lo, hi)
a.set_xlabel('Cross-validation macro-F1'); a.set_ylabel('Test macro-F1 (5-seed mean)')
a.set_title('CV ranking transfers to test', loc='left'); a.set_aspect('equal', adjustable='box')
for mname, r_ in T2.iterrows():
    b.scatter(r_.latency_ms, r_.test_f1, s=8 + 3.2 * r_.params_M, color=GC[r_.grp], alpha=0.85, edgecolor='black' if mname == SEL else 'white', lw=1 if mname == SEL else 0.4, zorder=3)
for mname, dx, dy in [('PVTv2-B2', 1.0, 1.045), ('DeiT III-S', 0.95, 1.06), ('MaxViT-T', 0.85, 1.045), ('MobileNetV2', 1.0, 1.06)]:
    r_ = T2.loc[mname]
    b.annotate(nm(mname), (r_.latency_ms, r_.test_f1), xytext=(r_.latency_ms * dx, r_.test_f1 * dy), fontsize=9, arrowprops=dict(arrowstyle='-', lw=0.5, color='0.4'), ha='center')
b.set_xscale('log'); b.set_xticks([5, 10, 20, 30], ['5', '10', '20', '30']); b.minorticks_off()
b.set_xlabel('GPU latency (ms, T4, batch = 1; log)'); b.set_ylabel('Test macro-F1 (5-seed mean)')
b.set_title('High accuracy achievable at low latency', loc='left'); b.set_ylim(0.5, 0.93); b.margins(x=0.08)
for g, c in GC.items():
    b.scatter([], [], color=c, s=18, label=g)
b.legend(frameon=False, loc='lower right', fontsize=9)
panel_letter(a, 'a', dx=-0.2, dy=1.03); panel_letter(b, 'b', dx=-0.2, dy=1.03)

# Fix ViT-S/16 annotation
for ch in list(b.get_children()):
    if isinstance(ch, mpl.text.Annotation) and ch.get_text() == 'ViT-S/16':
        ch.remove()
r_ = T2.loc['ViT-S/16']
b.annotate('ViT-S/16', (r_.latency_ms, r_.test_f1), xytext=(10, -20), textcoords='offset points', fontsize=9, ha='center', va='center', arrowprops=dict(arrowstyle='-', lw=0.5, color='0.4', shrinkA=1, shrinkB=4))
fig.savefig('figures/Fig5_cv_test_efficiency.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
print('a', a.get_position().bounds); print('b', b.get_position().bounds)
