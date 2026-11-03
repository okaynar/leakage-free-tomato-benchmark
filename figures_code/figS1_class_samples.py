# Run from the repository root:  python figures_code/figS1_class_samples.py
# Set TOMATO_ROOT to the dataset folder where raw images are needed.
import os
os.makedirs('figures', exist_ok=True)
import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

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


# Build dataframe of images
R = os.environ.get('TOMATO_ROOT', 'data/Tomato Dataset')
recs = []
for dp, dn, fn in os.walk(R):
    for f in fn:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            rel = os.path.relpath(dp, R).split(os.sep)
            recs.append((rel, f, os.path.join(dp, f)))

df = pd.DataFrame([{'top': r[0], 'split': r[1] if len(r) == 3 else 'raw', 'cls': r[-1], 'file': f, 'path': p} for r, f, p in recs])
df['lp'] = '\\\\?\\' + df.path

apply_figure_style(sizes=(10, 9, 9))
_tnr()

names = {'Tomato_Early_Blight': 'Early blight', 'Tomato_Healthy': 'Healthy', 'Tomato_leaf_late_blight': 'Late blight',
         'Tomato_leaf_yellow_curl_virus': 'TYLCV', 'Tomato_mold_leaf': 'Leaf mold',
         'Tomato_septora_leaf_spot': 'Septoria leaf spot',
         'Tomato_Early_blight_leaf': 'Early blight', 'Tomato_Healthy_leaf': 'Healthy',
         'Tomato_Septoria_leaf_spot': 'Septoria leaf spot'}
df['c'] = df.cls.map(names)
order = ['Early blight', 'Late blight', 'Septoria leaf spot', 'Leaf mold', 'TYLCV', 'Healthy']

Rw = df[df.top == 'Dataset (raw)']

rng = np.random.default_rng(3)
fig2, ax2 = plt.subplots(6, 5, figsize=(6.5, 8))
for r, o in enumerate(order):
    pick = Rw[Rw.c == o].sample(5, random_state=r).lp
    for k, p in enumerate(pick):
        a = ax2[r, k]
        im = Image.open(p)
        im.thumbnail((300, 300))
        a.imshow(im)
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    ax2[r, 0].set_ylabel(o, fontsize=9)
fig2.tight_layout()
fig2.savefig('figures/FigS1_class_samples.png', dpi=600, bbox_inches='tight')