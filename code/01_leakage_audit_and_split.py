"""
Leakage audit and source-group-disjoint split for the Mendeley tomato leaf dataset
(Malik, Ali & Ali, 2026; doi:10.17632/3mbnb82mxd.2).

Reproduces splits/raw_clean_group_split.csv from the raw subset ("Dataset (raw)").

Criteria for linking two images into the same SOURCE GROUP (union-find):
  (i)   identical MD5 file hash
  (ii)  dihedral-invariant 64-bit DCT pHash, min Hamming distance over the 8 D4 transforms <= 10
  (iii) DINOv2 ViT-S/14 (timm: vit_small_patch14_dinov2.lvd142m) class-token cosine similarity >= 0.88
  (iv)  same capture session: same source, |dt| <= 10 s (camera / epoch-ms names) or <= 2 WhatsApp
        message indices, AND cosine similarity >= 0.80
Groups containing more than one class are label conflicts and are removed entirely;
MD5 duplicates within a class keep the first copy. The clean set is split with
StratifiedGroupKFold(5, shuffle=True, random_state=42): the first outer fold is the test set,
the remainder is split again into 5 group-disjoint, class-stratified CV folds.

Usage:  python 01_leakage_audit_and_split.py --root "/path/to/Tomato Dataset" --out splits/
Requires: numpy pandas pillow scipy scikit-learn torch timm
"""
import argparse, hashlib, os, re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from sklearn.model_selection import StratifiedGroupKFold

FOLDER_TO_CLASS = {
    'Tomato_Early_Blight': 'Early blight', 'Tomato_Early_blight_leaf': 'Early blight',
    'Tomato_Healthy': 'Healthy', 'Tomato_Healthy_leaf': 'Healthy',
    'Tomato_leaf_late_blight': 'Late blight',
    'Tomato_leaf_yellow_curl_virus': 'Yellow leaf curl virus',
    'Tomato_mold_leaf': 'Leaf mold',
    'Tomato_septora_leaf_spot': 'Septoria leaf spot', 'Tomato_Septoria_leaf_spot': 'Septoria leaf spot',
}
CLASSES = ['Early blight', 'Late blight', 'Septoria leaf spot', 'Leaf mold', 'Yellow leaf curl virus', 'Healthy']
POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def list_raw_images(root):
    raw_dir = os.path.join(root, 'Dataset (raw)')
    recs = []
    for dp, _, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                p = os.path.join(dp, f)
                folder = os.path.basename(dp)
                recs.append(dict(path=p, file=f, folder=folder, c=FOLDER_TO_CLASS[folder],
                                 relpath=os.path.relpath(p, root).replace('\\', '/')))
    return pd.DataFrame(recs)


def dihedral_phash(path):
    """8 x 64-bit pHashes (4 rotations x mirror) of the 32x32 grayscale image."""
    with Image.open(path) as im:
        w, h = im.size
        im.draft('L', (128, 128))
        g = np.asarray(im.convert('L').resize((32, 32), Image.BILINEAR), dtype=np.float32)
    hs = []
    for k in range(4):
        for flip in (False, True):
            a = np.rot90(g, k)
            a = a[:, ::-1] if flip else a
            d = dctn(a, norm='ortho')[:8, :8].ravel()
            hs.append(np.packbits(d > np.median(d[1:])).view('>u8')[0])
    return w, h, hs


def hamming_min(H):
    """Pairwise min Hamming distance between hash 0 of image i and all 8 hashes of image j."""
    x = H[:, 0][:, None, None] ^ H[None]
    return POPCOUNT[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1).min(-1)


def dinov2_embeddings(paths, batch=32):
    import torch, timm
    m = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=224).eval()
    cfg = timm.data.resolve_data_config({}, model=m)
    cfg['input_size'] = (3, 224, 224); cfg['crop_pct'] = 1.0
    tf = timm.data.create_transform(**cfg)
    E = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[i:i + batch]])
            E.append(torch.nn.functional.normalize(m(x), dim=1).numpy())
    return np.vstack(E)


def capture_time(fname):
    m = re.match(r'IMG_(\d{8})_(\d{6})', fname)
    if m: return 'cam', datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S').timestamp()
    m = re.match(r'IMG-(\d{8})-WA(\d+)', fname)
    if m: return 'wa_' + m.group(1), int(m.group(2))
    m = re.match(r'(1[67]\d{11})', fname)
    if m: return 'ms', int(m.group(1)) / 1000
    return None, None


def build_groups(df, D, S, phash_max=10, cos_dup=0.88, gap_s=10, gap_wa=2, cos_session=0.80):
    n = len(df); parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    for a, b in zip(*np.where(np.triu((D <= phash_max) | (S >= cos_dup), 1))):
        union(a, b)
    for _, g in df.groupby('md5'):
        ii = g.index.tolist()
        for k in ii[1:]: union(ii[0], k)
    for src, g in df.dropna(subset=['src']).groupby('src'):
        ii, tt = g.index.values, g.t.values
        gap = gap_s if src in ('cam', 'ms') else gap_wa
        for x in range(len(ii)):
            for y in range(x + 1, len(ii)):
                if abs(tt[x] - tt[y]) <= gap and S[ii[x], ii[y]] >= cos_session:
                    union(ii[x], ii[y])
    return np.array([find(i) for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='folder that contains "Dataset (raw)"')
    ap.add_argument('--out', default='splits')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = list_raw_images(args.root).reset_index(drop=True)
    with ThreadPoolExecutor(8) as ex:
        ph = list(ex.map(dihedral_phash, df.path))
    df['w'] = [r[0] for r in ph]; df['h'] = [r[1] for r in ph]
    H = np.array([r[2] for r in ph], dtype=np.uint64)
    df['md5'] = df.path.map(lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest())

    D = hamming_min(H); np.fill_diagonal(D, 99)
    E = dinov2_embeddings(df.path.tolist()); S = E @ E.T; np.fill_diagonal(S, -1)
    df['src'], df['t'] = zip(*df.file.map(capture_time))

    df['group'] = build_groups(df, D, S)
    multi = df.groupby('group').c.nunique()
    df['label_conflict'] = df.group.isin(multi[multi > 1].index)
    df['md5_dup'] = df.duplicated(['md5', 'c'], keep='first')
    clean = df[~df.label_conflict & ~df.md5_dup].copy()

    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    _, te = next(sgk.split(clean, clean.c, clean.group))
    clean['split'] = 'trainval'; clean.iloc[te, clean.columns.get_loc('split')] = 'test'
    clean['fold'] = -1
    tv = np.where(clean.split == 'trainval')[0]
    for k, (_, v) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42)
                               .split(clean.iloc[tv], clean.c.iloc[tv], clean.group.iloc[tv])):
        clean.iloc[tv[v], clean.columns.get_loc('fold')] = k
    assert clean.groupby('group').split.nunique().max() == 1
    assert clean[clean.fold >= 0].groupby('group').fold.nunique().max() == 1

    clean['source_type'] = (clean.src.fillna('web/other').str.replace(r'wa_.*', 'whatsapp', regex=True)
                            .replace({'cam': 'camera_2025-11-02', 'ms': 'epoch_ms_name'}))
    clean['label'] = clean.c.map({k: i for i, k in enumerate(CLASSES)})
    out = clean.rename(columns={'c': 'class'})[['relpath', 'class', 'label', 'group', 'split', 'fold', 'md5', 'source_type']]
    out.to_csv(os.path.join(args.out, 'raw_clean_group_split.csv'), index=False)
    print('clean images:', len(out), '| groups:', out.group.nunique(), '| test:', (out.split == 'test').sum())


if __name__ == '__main__':
    main()
