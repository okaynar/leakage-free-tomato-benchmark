# Tomato leaf disease classification under a leakage-free (source-group-disjoint) protocol

This repository contains the **data splits, leakage-audit outputs and code** used in the study
*"Leakage-Aware Benchmarking of CNNs and Vision Transformers for Tomato Leaf Disease Classification under Real-Field Conditions"*
(manuscript submitted to Gümüşhane University Journal of Science and Technology).

The images themselves are **not** redistributed here. Download them from the original source:

> Malik, M. N., Ali, S. H., & Ali, H. (2026). *Tomato Leaf Disease Classification Dataset* (Version 2) [Data set]. Mendeley Data. https://doi.org/10.17632/3mbnb82mxd.2 (CC BY 4.0)

All file paths in this repository (`relpath`) are relative to the dataset folder `Tomato Dataset/`.

---

## 1. Why a new split?

The archive ships an *augmented, pre-split* subset (7,200 images, 1,200 per class). Our audit showed that augmentation was applied **before** splitting:
**325 / 720 test images (45%)** and **646 / 1,440 validation images (45%)** belong to the same source group as at least one training image
(`splits/presplit_subset_audit_manifest.csv`). That split is therefore not used. All experiments use the **raw subset** (830 images),
audited and re-split at the **source-group level**.

## Quick start

| What you want to do | GPU? | Command / notebook | Needs the images? |
|---|---|---|---|
| Use the leakage-free split in your own experiments | no | read `splits/raw_clean_group_split.csv` | yes (for training) |
| Redraw every figure of the paper from the shipped results | no | `python figures_code/<script>.py` | only Fig. 7, 8, S1, S5 |
| Recompute ensemble / McNemar / bootstrap from the stored test probabilities | no | load `results/test_probs/*.npy` (see §6) | no |
| Regenerate the split from the raw images (leakage audit) | no (CPU is enough) | `python code/01_leakage_audit_and_split.py` | yes |
| Re-train all 20 models (CV + final + test + latency) | yes (T4) | `notebooks/02_train_cv_final_20models.ipynb` | yes |
| Re-train PVTv2-B2 / Swin-T, save weights, Grad-CAM | yes (T4) | `notebooks/03_final_weights_gradcam.ipynb` | yes |

```bash
git clone https://github.com/okaynar/tomato-leaf
cd tomato-leaf
pip install -r requirements.txt

# 1) figures from the shipped results (run from the repository root; output → figures/)
export TOMATO_ROOT="/path/to/Tomato Dataset"        # only for scripts that show images
for f in figures_code/*.py; do python "$f"; done

# 2) regenerate the split (writes splits_regenerated/raw_clean_group_split.csv)
python code/01_leakage_audit_and_split.py --root "/path/to/Tomato Dataset" --out splits_regenerated/

# 3) training notebooks locally (GPU recommended); new results go to runs/
export TOMATO_DATA="/path/to/Tomato Dataset"
export TOMATO_SMOKE=1            # optional: end-to-end check on CPU (≈ 7 min) (tiny models, no pretrained weights)
jupyter lab notebooks/
```

**Google Colab:** upload the unzipped `Tomato Dataset` folder to `MyDrive/Tomato Dataset`, open the notebook in Colab, select a T4 GPU, and add a
first cell `!git clone https://github.com/okaynar/tomato-leaf /content/tomato-leaf`. Cell *2) Paths* is the only cell with paths
(`REPO_DIR`, `DATA_ROOT`, `OUT_DIR`); new results are written to `MyDrive/tomato-leaf-runs`, never into the repository's `results/`.
Notebook 02 resumes after a disconnect by skipping finished (model, seed, fold) runs. The paper's full run took ≈ 16 h of training on a T4
(13.8 h CV + 1.9 h final; `minutes` columns of `results/cv_runs.csv` and `results/final_test_runs.csv`). MambaVision-T needs the CUDA
`mamba-ssm` 2.3.2.post1 wheel; cell 0 pins torch 2.10.0 on Colab for this.

Tested before release: all 10 figure scripts run from a clean copy of the repository; notebooks 02 and 03 complete end to end in `TOMATO_SMOKE=1`
mode on CPU (≈ 7 and 5 min); the stored test probabilities reproduce the seed-mean and ensemble macro-F1 of all 20 models in `results/results_20models_cv_test.csv`.
Code comments inside the notebooks are in Turkish; each notebook starts with an English "How to run" cell.

## 2. Repository layout

```
tomato-leaf/
├── README.md
├── LICENSE                                # MIT (code); data-derived files stay CC BY 4.0
├── requirements.txt
├── splits/
│   ├── raw_clean_group_split.csv          # THE split used in the paper (748 images)
│   ├── excluded_images.csv                # 82 raw images removed, with reason
│   ├── label_conflicts.csv                # 47 images in 12 multi-class source groups
│   ├── split_summary.csv                  # per-class counts per split / fold
│   ├── dataset_summary.csv                # raw vs augmented subset counts per class
│   └── presplit_subset_audit_manifest.csv # audit of the original augmented 7,200-image split
├── code/
│   └── 01_leakage_audit_and_split.py      # MD5 + dihedral pHash + DINOv2 + capture session → groups → split
├── notebooks/
│   ├── 02_train_cv_final_20models.ipynb   # 20 models: 5-fold × 5-seed CV, final training, test, efficiency (Google Colab)
│   └── 03_final_weights_gradcam.ipynb     # PVTv2-B2 / Swin-T final weights, Grad-CAM, deletion test (Google Colab)
├── figures_code/                          # scripts that draw every figure of the paper
└── results/
    ├── cv_runs.csv                        # 500 CV runs (model, seed, fold, best epoch, per-epoch val macro-F1, metrics)
    ├── final_test_runs.csv                # 100 final runs (20 models × 5 seeds) on the locked test set
    ├── results_20models_cv_test.csv       # per-model summary (CV, test, ensemble + CI, McNemar, efficiency)
    ├── pvtv2_per_class.csv                # per-class precision / recall / F1 of the selected model
    ├── gradcam_per_image_stats.csv        # per-test-image Grad-CAM statistics (150 images) + capture source
    ├── test_probs/<model>__seed<k>.npy    # softmax outputs of the 100 final models on the 150 test images (order = test rows of the split file)
    └── gradcam_v7/
        ├── gradcam_raw_7x7.npz            # raw 7×7 Grad-CAM maps (5 seeds × 2 models × 150 test images)
        ├── gradcam_per_image.csv          # same per-image table as written by notebook 03
        └── gradcam_deletion.csv           # deletion test: probability drop, Grad-CAM vs random regions, Wilcoxon p
```

## 3. The split files

### `splits/raw_clean_group_split.csv`

| Column | Meaning |
|---|---|
| `relpath` | image path relative to `Tomato Dataset/` |
| `class` | one of: Early blight, Late blight, Septoria leaf spot, Leaf mold, Yellow leaf curl virus, Healthy |
| `label` | integer label 0–5 in the order above |
| `group` | source-group id (union–find root index); all images of a group are always in the same split/fold |
| `split` | `test` (locked test set, used once) or `trainval` |
| `fold` | CV fold 0–4 for `trainval` images, −1 for `test` |
| `md5` | MD5 hash of the image file (to verify your copy) |
| `source_type` | capture source inferred from the file name: `camera_2025-11-02`, `whatsapp`, `epoch_ms_name`, `web/other` |

Per-class counts (`splits/split_summary.csv`):

| Class | Test | Train/val | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Total | Groups |
|---|---|---|---|---|---|---|---|---|---|
| Early blight | 21 | 84 | 17 | 16 | 17 | 17 | 17 | 105 | 100 |
| Late blight | 20 | 81 | 16 | 17 | 16 | 16 | 16 | 101 | 97 |
| Septoria leaf spot | 34 | 136 | 27 | 28 | 27 | 27 | 27 | 170 | 152 |
| Leaf mold | 37 | 146 | 29 | 29 | 30 | 29 | 29 | 183 | 161 |
| Yellow leaf curl virus | 15 | 62 | 13 | 12 | 12 | 12 | 13 | 77 | 72 |
| Healthy | 23 | 89 | 18 | 18 | 17 | 18 | 18 | 112 | 100 |
| **Total** | **150** | **598** | 120 | 120 | 119 | 119 | 120 | **748** | 682 |

Capture source by class (a known source–class confound; see the paper's Discussion):

| Class | camera_2025-11-02 | whatsapp | epoch_ms_name | web/other |
|---|---|---|---|---|
| Early blight | 33 | 12 | 13 | 47 |
| Healthy | 46 | 0 | 0 | 66 |
| Late blight | 5 | 1 | 0 | 95 |
| Leaf mold | 10 | 136 | 1 | 36 |
| Septoria leaf spot | 0 | 22 | 4 | 144 |
| Yellow leaf curl virus | 27 | 0 | 0 | 50 |

### `splits/excluded_images.csv`

The 830 raw images minus the 748 clean images: **82 images**, reason `md5_duplicate` (35), `label_conflict` (46) or both (1).
In total 36 exact duplicates (second and later copies within a class) and 47 label-conflict images were removed.

## 4. Leakage audit (code/01_leakage_audit_and_split.py)

Two raw images are linked into the same **source group** if **any** criterion holds; groups are the connected components (union–find):

| Criterion | Rule | Detects |
|---|---|---|
| Exact duplicate | identical MD5 | re-uploaded files |
| Dihedral pHash | 64-bit DCT pHash on 32×32 grayscale; minimum Hamming distance over the 8 rotations/mirrors (D4) ≤ 10 | rotated / flipped / re-encoded copies |
| Deep embedding | cosine similarity of DINOv2 ViT-S/14 (`timm: vit_small_patch14_dinov2.lvd142m`) class tokens ≥ 0.88 | crops, zooms, consecutive frames |
| Capture session | same source and Δt ≤ 10 s (camera, epoch-ms names) or ≤ 2 WhatsApp message indices, **and** cosine ≥ 0.80 | burst shots of the same leaf |

Thresholds were set from the data: cross-class pHash matches stay almost constant up to d ≤ 10 and rise sharply from d ≥ 12;
s ≥ 0.88 corresponded to crops / consecutive frames on visual inspection.

Then: groups with more than one class → removed (label conflicts); MD5 duplicates within a class → first copy kept;
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` → first outer fold = test (150 images, 137 groups),
remaining 598 images (545 groups) → 5 group-disjoint, class-stratified CV folds (120 / 120 / 119 / 119 / 120).

Leakage checks on the final split: no MD5 shared between test and train/val; minimum pHash distance between any test and train/val image = 12 (> 10);
maximum DINOv2 similarity = 0.877 (< 0.88). The training notebook re-checks the MD5/pHash conditions at the start of every run.

```bash
pip install -r requirements.txt
python code/01_leakage_audit_and_split.py --root "/path/to/Tomato Dataset" --out splits_regenerated/
```

Note: the script is the cleaned, standalone version of the code that produced the published split. The MD5 and pHash steps were re-run on the
original files and reproduce the published values (42 MD5 groups, 36 within-class copies, 65 pHash pairs, all same-group in the published split,
test–train minimum pHash distance 12). The DINOv2 step depends on the downloaded weights; `group` ids are union–find indices, so a regenerated file may
number groups differently while giving the same partition. **Use `splits/raw_clean_group_split.csv` to reproduce the paper.**

## 5. Training and evaluation protocol (notebooks/02_train_cv_final_20models.ipynb)

- **Models (20, ImageNet-1k weights only, no distillation):** C-CNN: ResNet50, InceptionV3, DenseNet121, MobileNetV2, EfficientNet-B0 ·
  M-CNN: ConvNeXt V2-Tiny, EfficientNetV2-S, InceptionNeXt-Tiny, MobileNetV4-Conv-M, GhostNetV2-1.0 ·
  ViT: ViT-S/16, DeiT III-S, Swin-T, PVTv2-B2, XCiT-S12/16 · Hybrid: CoAtNet-0, MaxViT-T, FastViT-SA12, MobileViTv2-1.0, MambaVision-T.
  19 models from `timm`; MambaVision-T from Hugging Face `nvidia/MambaVision-T-1K` (revision `b1de77e`, NVIDIA Source Code License-NC, non-commercial).
- **Common recipe:** 224×224 input; class-weighted cross-entropy with label smoothing ε = 0.1; AdamW (β1 = 0.9, β2 = 0.999, weight decay 0.05);
  learning rate 1e-4 (backbone) / 1e-3 (head); 3-epoch linear warm-up + cosine decay; batch 32; max 40 epochs; early stopping patience 7 on
  validation macro-F1; FP16 mixed precision. Train-only augmentation: RandomResizedCrop(224, scale 0.6–1), horizontal/vertical flip (p = 0.5),
  rotation ±20°, ColorJitter(0.2, 0.2, 0.2, 0.02). Validation/test: resize short side 224 + center crop.
- **Cross-validation:** 5 folds × 5 seeds (0–4) = 25 runs per model, 500 in total.
- **Model selection (CV data only):** Friedman test + Nemenyi CD over the 25 seed–fold blocks; the best-CV model is compared with all others by Wilcoxon;
  among models not significantly different, the one with the fewest parameters is selected.
- **Final training:** all 598 train/val images, epochs = median best CV epoch, 5 seeds; the 150-image test set is evaluated **once**.
  Seed mean ± SD and the 5-seed softmax-average ensemble are reported, with a 2,000-resample bootstrap 95% CI and Holm-corrected exact McNemar tests.
- **Hardware:** Google Colab, NVIDIA Tesla T4 (16 GB). Latency: batch 1, mean of 50 passes after 10 warm-up passes.

Notebook 03 retrains the five final PVTv2-B2 and Swin-T models (E = 10, seeds 0–4), saves fp16 weights, and computes Grad-CAM maps, seed stability,
cross-model agreement and a deletion test.

## 6. Main results (`results/results_20models_cv_test.csv`)

Example — recompute the ensemble macro-F1 of a model from the stored probabilities:

```python
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
split = pd.read_csv('splits/raw_clean_group_split.csv')
y = split[split.split == 'test'].label.values                       # 150 labels, same order as the .npy rows
P = np.mean([np.load(f'results/test_probs/PVTv2-B2__seed{s}.npy') for s in range(5)], axis=0)
print(f1_score(y, P.argmax(1), average='macro'))                    # 0.915
```
(Model names with "/" use "-" in file names, e.g. `XCiT-S12-16__seed0.npy`.)


The selected model is **PVTv2-B2**: CV macro-F1 0.855 ± 0.022, test macro-F1 0.880 ± 0.018 (5 seeds),
5-seed ensemble 0.915 (95% CI 0.862–0.958).

| Group | Model | Params (M) | Latency (ms, T4) | CV macro-F1 | Test macro-F1 (5 seeds) | Ensemble macro-F1 [95% CI] |
|---|---|---|---|---|---|---|
| ViT | Swin-T | 27.5 | 11.6 | 0.863 ± 0.026 | 0.866 ± 0.008 | 0.897 [0.841–0.947] |
| ViT | PVTv2-B2 | 24.9 | 13.4 | 0.855 ± 0.022 | 0.880 ± 0.018 | 0.915 [0.862–0.958] |
| Hybrid | CoAtNet-0 | 26.7 | 11.0 | 0.850 ± 0.023 | 0.863 ± 0.020 | 0.869 [0.804–0.923] |
| Hybrid | MambaVision-T | 31.2 | 11.7 | 0.849 ± 0.024 | 0.853 ± 0.014 | 0.887 [0.827–0.936] |
| ViT | DeiT III-S | 21.7 | 6.2 | 0.847 ± 0.015 | 0.860 ± 0.013 | 0.890 [0.827–0.938] |
| ViT | XCiT-S12/16 | 25.9 | 16.2 | 0.846 ± 0.028 | 0.859 ± 0.040 | 0.883 [0.823–0.936] |
| M-CNN | ConvNeXtV2-Tiny | 27.9 | 9.0 | 0.846 ± 0.030 | 0.866 ± 0.021 | 0.868 [0.805–0.921] |
| ViT | ViT-S/16 | 21.7 | 5.7 | 0.843 ± 0.024 | 0.858 ± 0.039 | 0.900 [0.843–0.946] |
| Hybrid | MaxViT-T | 30.4 | 27.5 | 0.839 ± 0.018 | 0.851 ± 0.021 | 0.867 [0.803–0.921] |
| M-CNN | InceptionNeXt-Tiny | 25.8 | 7.9 | 0.837 ± 0.034 | 0.795 ± 0.024 | 0.788 [0.713–0.851] |
| Hybrid | MobileViTv2-1.0 | 4.4 | 9.7 | 0.834 ± 0.022 | 0.783 ± 0.020 | 0.793 [0.716–0.856] |
| C-CNN | DenseNet121 | 7.0 | 17.4 | 0.819 ± 0.028 | 0.804 ± 0.007 | 0.826 [0.757–0.886] |
| Hybrid | FastViT-SA12 | 10.6 | 11.1 | 0.815 ± 0.027 | 0.778 ± 0.015 | 0.793 [0.721–0.854] |
| M-CNN | GhostNetV2-1.0 | 4.9 | 14.6 | 0.800 ± 0.024 | 0.738 ± 0.029 | 0.726 [0.645–0.791] |
| C-CNN | ResNet50 | 23.5 | 6.8 | 0.768 ± 0.033 | 0.792 ± 0.014 | 0.812 [0.741–0.872] |
| C-CNN | InceptionV3 | 21.8 | 13.1 | 0.700 ± 0.036 | 0.679 ± 0.029 | 0.700 [0.618–0.772] |
| M-CNN | EfficientNetV2-S | 20.2 | 18.4 | 0.683 ± 0.039 | 0.655 ± 0.021 | 0.778 [0.700–0.840] |
| M-CNN | MobileNetV4-Conv-M | 8.4 | 8.5 | 0.642 ± 0.047 | 0.579 ± 0.042 | 0.714 [0.632–0.783] |
| C-CNN | EfficientNet-B0 | 4.0 | 8.3 | 0.625 ± 0.041 | 0.600 ± 0.040 | 0.709 [0.627–0.780] |
| C-CNN | MobileNetV2 | 2.2 | 5.8 | 0.617 ± 0.035 | 0.547 ± 0.022 | 0.701 [0.619–0.768] |

## 7. Figures

`figures_code/` contains the scripts for every figure (run from the repository root; output goes to `figures/`):
`fig1_pipeline.py`, `fig2_cleaning_sources.py`, `fig3_cv_results.py`, `fig4_ensemble_ci.py`, `fig5_cv_test_efficiency.py`, `fig6_pvtv2_confusion.py`,
`fig7_fig8_figS5_gradcam.py`, `fig9_convergence.py`, `figS1_class_samples.py`, `figS4_cv_protocol.py`.
Outputs: `Fig1_pipeline.png` … `Fig9_convergence.png`, `FigS1_class_samples.png`, `FigS4_cv_protocol.png`, `FigS5_gradcam_errors.png`.
Scripts that show images (Fig. 7, Fig. 8, S1, S5) read the dataset folder from the `TOMATO_ROOT` environment variable; all others need only the
files in `results/` and `splits/`. Times New Roman is used if installed, otherwise Liberation Serif / DejaVu Serif.

## 8. Licences

- Dataset: CC BY 4.0 (Malik et al., 2026) — cite the dataset DOI when using the split files.
- MambaVision-T weights/code: NVIDIA Source Code License-NC (non-commercial use only).
- Code in this repository (`code/`, `notebooks/`, `figures_code/`): MIT licence (see `LICENSE`).

## 9. Citation

[Paper citation to be added after publication.]
