# Run from the repository root:  python figures_code/fig2_cleaning_sources.py
# Set TOMATO_ROOT to the dataset folder where raw images are needed.
import os
os.makedirs('figures', exist_ok=True)
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch

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


m = pd.read_csv('splits/raw_clean_group_split.csv')

order_en = ['Early blight', 'Late blight', 'Septoria leaf spot', 'Leaf mold', 'Yellow leaf curl virus', 'Healthy']
tr = {'Early blight': 'Early\nblight', 'Late blight': 'Late\nblight', 'Septoria leaf spot': 'Septoria\nleaf spot', 'Leaf mold': 'Leaf\nmold', 'Yellow leaf curl virus': 'TYLCV', 'Healthy': 'Healthy'}
ct = pd.crosstab(m.source_type, m['class'])[order_en].loc[['camera_2025-11-02', 'whatsapp', 'epoch_ms_name', 'web/other']]
pct = ct / ct.sum() * 100

apply_figure_style(sizes=(10, 9, 9))
_tnr()
fig = plt.figure(figsize=(6.7, 2.9))
sfa, sfb = fig.subfigures(1, 2, width_ratios=[1, 1.7], wspace=0.04)
a1 = sfa.add_axes([0.24, 0.20, 0.75, 0.66])
x = np.arange(5); h = [830, 36, 47, 748, 682]; bottom = [0, 794, 747, 0, 0]; cols = ['#4C72B0', '#C44E52', '#C44E52', '#4C72B0', '#8C8C8C']
a1.bar(x, h, bottom=bottom, color=cols, width=0.62)
for i in range(5): a1.text(i, bottom[i] + h[i] + 12, str(h[i]), ha='center', fontsize=9)
a1.set_xticks(x, ['Raw', 'MD5\ndupl.', 'Label\nconfl.', 'Clean', 'Indep.\ngroups'], fontsize=9)
a1.tick_params(axis='x', length=0)
a1.set_ylabel('Image count'); a1.set_ylim(0, 900); a1.set_yticks([0, 200, 400, 600, 800])
sfa.text(0.02, 0.975, 'a', fontweight='bold', fontsize=11, va='top')
sfa.text(0.11, 0.975, 'Cleaning of the raw dataset', fontsize=10, va='top')
a2 = sfb.add_axes([0.20, 0.20, 0.665, 0.66])
im = a2.imshow(pct.values, cmap='Blues', vmin=0, vmax=100, aspect='auto')
for i in range(4):
    for j in range(6): a2.text(j, i, str(ct.values[i, j]), ha='center', va='center', fontsize=9, color='white' if pct.values[i, j] > 55 else 'black')
a2.set_xticks(range(6), [tr[o] for o in order_en], fontsize=9)
a2.set_yticks(range(4), ['Camera\n(2 Nov 2025)', 'WhatsApp\n(Jul\u2013Aug 2024)', 'Epoch-ms\nfilename', 'Web/other'], fontsize=9)
a2.tick_params(length=0)
for sp in a2.spines.values(): sp.set_visible(False)
cax = sfb.add_axes([0.885, 0.20, 0.022, 0.66])
cb = fig.colorbar(im, cax=cax); cb.set_label('Share within class (%)', fontsize=9); cb.ax.tick_params(labelsize=9); cb.outline.set_linewidth(0.5)
sfb.text(0.01, 0.975, 'b', fontweight='bold', fontsize=11, va='top')
sfb.text(0.06, 0.975, 'Image source varies by class', fontsize=10, va='top')
fig.savefig('figures/Fig2_cleaning_sources.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
print(ct.sum().to_dict(), ct.sum().sum())
