"""Draw a random MEETI sample list for the CAM pipeline.

    python -m visualization.attention_grad_cam.make_meeti_manifest --n 30 --seed 42 --split train

Split membership and label existence come from the 96/2/2 split CSVs
(<split>.csv); signal/image paths come from the local
MEETI tree. target_text is left empty on purpose: SGL reads report and
LLM_Interpretation straight out of each .mat.
"""
import argparse
import csv
from pathlib import Path

from visualization.grad_cam.checkpoint_io import LABEL_ROOT

DEFAULT_DATA_ROOT = ' ./datasets/MEETI'
META_COLS = ('subject_id', 'study_id', 'total_report')

def parser():
    p = argparse.ArgumentParser(description='Random MEETI manifest for Grad-CAM.')
    p.add_argument('--n', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    p.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    p.add_argument('--label-root', default=str(LABEL_ROOT))
    p.add_argument('--out', default=str(Path(__file__).parent / 'meeti_manifest.csv'))
    return p

def main(argv=None):
    args = parser().parse_args(argv)
    import numpy as np

    csv_path = Path(args.label_root) / f'{args.split}.csv'
    if not csv_path.is_file():
        raise FileNotFoundError(f'Split label CSV not found: {csv_path}')
    root = Path(args.data_root)
    if not root.is_dir():
        raise FileNotFoundError(f'MEETI root not found: {root}')

    # Pass 1: split membership (study_id column only; no float parsing).
    ids = []
    with csv_path.open(encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f)
        head = next(rd)
        col = head.index('study_id')
        for row in rd:
            ids.append(row[col])
    print(f'{args.split}: {len(ids)} rows in the split CSV')

    # Local tree: one .mat + one .png per study directory.
    mats = {}
    for m in root.rglob('*.mat'):
        mats.setdefault(m.stem, m)
    eligible = [i for i in ids if i in mats and mats[i].with_suffix('.png').is_file()]
    print(f'  {len(mats)} studies in {root}, {len(eligible)} of them are in this split with signal+image')
    if args.n > len(eligible):
        raise ValueError(f'--n {args.n} exceeds {len(eligible)} eligible samples')

    rng = np.random.default_rng(args.seed)
    picked = sorted(rng.choice(len(eligible), size=args.n, replace=False).tolist())
    chosen = [eligible[i] for i in picked]
    wanted = set(chosen)

    # Pass 2: ground-truth positive counts for the drawn samples only.
    pos = {}
    with csv_path.open(encoding='utf-8-sig', newline='') as f:
        rd = csv.DictReader(f)
        label_cols = [c for c in rd.fieldnames if c not in META_COLS]
        for row in rd:
            if row['study_id'] in wanted:
                pos[row['study_id']] = sum(int(float(row[c] or 0)) for c in label_cols)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'dataset', 'signal_path', 'image_path',
                    'target_class', 'target_text', 'task'])
        for sid in chosen:
            mat = mats[sid]
            w.writerow([sid, 'meeti', str(mat), str(mat.with_suffix('.png')), '', '', ''])

    counts = [pos.get(i, 0) for i in chosen]
    print(f'{len(chosen)} MEETI {args.split} samples -> {out}')
    print('  ids:', ' '.join(chosen))
    print('  ground-truth positive labels per sample:', counts, '(sum=%d)' % sum(counts))
    print(f'  (seed={args.seed}, split={args.split}, eligible={len(eligible)})')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
