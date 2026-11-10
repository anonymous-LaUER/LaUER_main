"""Draw a random CPSC2018 (icbeb) sample list for the CAM pipeline.

    python -m visualization.attention_grad_cam.make_cpsc_manifest --n 30 --seed 42 --split train

Labels are cpsc2018's own 9 classes (zeroshot/data/tasks.py), NOT the MEETI bank.
Signal/image paths are resolved from the split CSV's relative paths against the raw CPSC root.
CPSC has no report field, so the SGL alignment text is built from the sample's own
positive labels, expanded through the same frozen prompt snapshot as the queries.
"""
import argparse
import csv
from pathlib import Path

from visualization.grad_cam.checkpoint_io import CPSC_SPLIT_ROOT, CPSC_PROMPT_PATH
from .make_csn_manifest import stratified_pick

DEFAULT_ROOT = 'datasets/CPSC2018'

def parser():
    p = argparse.ArgumentParser(description='Random CPSC2018 manifest for Grad-CAM.')
    p.add_argument('--n', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    p.add_argument('--root', default=DEFAULT_ROOT, help='Raw CPSC2018 root')
    p.add_argument('--stratify', dest='stratify', action='store_true', default=True,
                   help='Allocate the draw across label_count strata (default on). CPSC is 89%% '
                        'single-label, so a uniform draw barely exercises the multi-label path.')
    p.add_argument('--no-stratify', dest='stratify', action='store_false')
    p.add_argument('--out', default=str(Path(__file__).parent / 'cpsc_manifest.csv'))
    p.add_argument('--prompt-path', default=str(CPSC_PROMPT_PATH),
                   help='Prompt snapshot used for the query texts and the target_text column')
    return p

def main(argv=None):
    args = parser().parse_args(argv)
    import numpy as np
    from zeroshot.data.tasks import TASKS
    from zeroshot.prompts import prompts_for_labels

    labels = list(TASKS['cpsc2018'].labels)
    phrases = dict(zip(labels, prompts_for_labels(labels, args.prompt_path, task='CPSC')))
    split_csv = Path(CPSC_SPLIT_ROOT) / f'{args.split}.csv'
    if not split_csv.is_file():
        raise FileNotFoundError(f'Split CSV not found: {split_csv}')
    root = Path(args.root)

    rows, positives = {}, {}
    with split_csv.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            sid = row['sample_id']
            rows[sid] = row
            positives[sid] = int(row.get('label_count') or 0)

    def resolve(sid, key):
        rel = (rows[sid].get(key) or '').strip()
        return str(root / rel) if rel else ''

    ids = [s for s in rows if positives.get(s, 0) > 0
           and resolve(s, 'signal_path') and resolve(s, 'image_path')]
    missing = [s for s in ids if not Path(resolve(s, 'signal_path')).is_file()
               or not Path(resolve(s, 'image_path')).is_file()]
    if missing:
        print(f'  dropped {len(missing)} samples whose files are not on disk, e.g. {missing[:3]}')
        ids = [s for s in ids if s not in set(missing)]
    print(f'{args.split}: {len(rows)} rows in the split CSV, {len(ids)} with signal+image on disk')
    if args.n > len(ids):
        raise ValueError(f'--n {args.n} exceeds {len(ids)} eligible samples')

    rng = np.random.default_rng(args.seed)
    if args.stratify:
        chosen = stratified_pick(ids, positives, args.n, rng)
    else:
        chosen = sorted(ids[i] for i in rng.choice(len(ids), size=args.n, replace=False).tolist())

    def alignment_text(sid):
        return '. '.join(phrases[n] for n in labels if str(rows[sid].get(n, '0')) == '1')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'dataset', 'signal_path', 'image_path',
                    'target_class', 'target_text', 'task'])
        for sid in chosen:
            w.writerow([sid, 'cpsc2018', resolve(sid, 'signal_path'), resolve(sid, 'image_path'),
                        '', alignment_text(sid), 'cpsc2018'])

    counts = [positives[i] for i in chosen]
    print(f'{len(chosen)} CPSC {args.split} samples -> {out}')
    print('  ids:', ' '.join(chosen))
    print('  label_count per sample:', counts, '(sum=%d)' % sum(counts))
    print(f'  target_text example ({chosen[0]}): {alignment_text(chosen[0])[:130]}')
    print(f'  (seed={args.seed}, stratify={args.stratify}, '
          f'pool mean label_count={sum(positives[i] for i in ids)/len(ids):.2f})')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
