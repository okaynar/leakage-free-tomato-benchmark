# Run from the repository root:  python figures_code/fig1_pipeline.py
# Set TOMATO_ROOT to the dataset folder where raw images are needed.
import os
os.makedirs('figures', exist_ok=True)
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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


plt.rcdefaults()
apply_figure_style(sizes=(10, 9, 9))
_tnr()

boxes = [
    ('1. Raw data', ['830 images, 6 classes', '150–4,416 px', 'Camera / WhatsApp / web', '(pre-augmented split', 'not used)'], '#E8EEF7'),
    ('2. Leakage audit', ['MD5 (exact duplicate)', 'Dihedral pHash, d ≤ 10', 'DINOv2 cosine ≥ 0.88', 'Capture-session link', '→ union–find groups'], '#FBE8DF'),
    ('3. Cleaning', ['−36 MD5 duplicates', '−47 label conflicts', '(1 shared image)', '= 748 images', '682 independent groups'], '#FBE8DF'),
    ('4. Group-disjoint split', ['StratifiedGroupKFold', 'Test: 150 (137 groups)', 'Train/val: 598', '→ 5 folds (545 groups)', 'Leakage check (assert)'], '#E8F2E8'),
    ('5. Training', ['20 models (4 groups)', 'ImageNet-1k weights', '5 folds × 5 seeds', '= 500 CV runs', 'Train-only augmentation', 'AdamW + cosine, AMP'], '#F1ECF7'),
    ('6. Model selection', ['CV data only', 'Mean macro-F1', 'Friedman + Nemenyi CD', 'Wilcoxon (α = 0.05)', 'Ties broken by', 'fewest parameters'], '#F2F2F2'),
    ('7. Final training', ['Each model: 598 images', '5 seeds', 'Epochs = median of', 'best CV epochs', '(no validation set)'], '#F1ECF7'),
    ('8. Test (single pass)', ['150 images', 'Macro-F1, MCC, κ', 'Seed mean ± SD', 'Seed ensemble', 'Bootstrap 95% CI', 'McNemar, efficiency'], '#F2F2F2'),
]

fig, ax = plt.subplots(figsize=(6.7, 2.95))
ax.set_xlim(0, 4)
ax.set_ylim(0, 2)
ax.axis('off')

pos = [(0, 1), (1, 1), (2, 1), (3, 1), (3, 0), (2, 0), (1, 0), (0, 0)]
W, H = 0.94, 0.90

for (t, lines, c), (cx, cy) in zip(boxes, pos):
    x0, y0 = cx + 0.03, cy + 0.05
    ax.add_patch(FancyBboxPatch((x0, y0), W, H, boxstyle='round,pad=0,rounding_size=0.04', fc=c, ec='0.35', lw=1))
    ax.text(x0 + W / 2, y0 + H - 0.085, t, ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(x0 + W / 2, y0 + H - 0.17, '\n'.join(lines), ha='center', va='top', fontsize=9, linespacing=1.22)

arr = dict(arrowstyle='-|>', color='0.3', lw=1.2, mutation_scale=9)
for i in range(3):
    ax.annotate('', xy=(i + 1.035, 1.5), xytext=(i + 0.965, 1.5), arrowprops=arr)
ax.annotate('', xy=(3.5, 0.955), xytext=(3.5, 1.045), arrowprops=arr)
for i in [3, 2, 1]:
    ax.annotate('', xy=(i - 0.035, 0.5), xytext=(i + 0.035, 0.5), arrowprops=arr)

fig.subplots_adjust(0, 0, 1, 1)
fig.savefig('figures/Fig1_pipeline.png', dpi=600, bbox_inches='tight', pad_inches=0.02)