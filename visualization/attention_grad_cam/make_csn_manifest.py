"""Draw a random CSN (chapman) sample list for the CAM pipeline.

    python -m visualization.attention_grad_cam.make_csn_manifest --n 10 --seed 42

Writes a manifest readable by visualization.attention_grad_cam.run --manifest.
Signal/image paths come from the same resolver the linear probes use, so the CAM sees
exactly the files the downstream heads were trained on.
"""
import argparse
import csv
from pathlib import Path

from visualization.grad_cam.checkpoint_io import ROOT, CSN_SPLIT_ROOT, CSN_PROMPT_PATH

def parser():
    p = argparse.ArgumentParser(description='Random CSN manifest for Grad-CAM.')
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--require-positive', action='store_true', default=True,
                   help='Keep only samples with at least one positive CSN label (default on)')
    p.add_argument('--allow-empty', dest='require_positive', action='store_false')
    p.add_argument('--stratify', dest='stratify', action='store_true', default=True,
                   help='Allocate the draw across label_count strata proportionally (default on)')
    p.add_argument('--no-stratify', dest='stratify', action='store_false',
                   help='Plain uniform draw. Note: 62.8%% of the CSN test split is single-label, '
                        'so a uniform n=10 draw comes out all-single-label about 1%% of the time.')
    p.add_argument('--out', default=str(Path(__file__).parent / 'csn_manifest.csv'))
    p.add_argument('--exclude', nargs='+', default=[],
                   help='Manifest CSVs whose sample_ids must not be drawn again')
    p.add_argument('--prompt-path', default=str(CSN_PROMPT_PATH),
                   help='Prompt snapshot used to fill the target_text column (SGL alignment)')
    p.add_argument('--no-target-text', dest='target_text', action='store_false', default=True,
                   help='Leave target_text empty; SGL then needs an explicit --target-text')
    p.add_argument('--csn-root', default='datasets/CSN',
                   help='Raw CSN root; CSN_index_with_images.csv supplies the report-equivalent text')
    return p

def stratified_pick(ids, positives, n, rng):
    """Allocate n across label_count strata proportionally, then draw inside each stratum.

    A plain uniform draw is a bad default here: 62.8% of the CSN test split carries a single
    label, so an all-single-label set is not rare, and it makes the per-label CAM export
    useless for checking whether different labels actually select different regions.
    """
    strata = {}
    for i in ids:
        strata.setdefault(positives.get(i, 0), []).append(i)
    keys = sorted(strata)
    total = len(ids)
    quota = {k: n * len(strata[k]) / total for k in keys}
    alloc = {k: min(int(quota[k]), len(strata[k])) for k in keys}
    while sum(alloc.values()) < n:
        spare = [k for k in keys if alloc[k] < len(strata[k])]
        if not spare:
            raise ValueError(f'Only {sum(alloc.values())} samples available for --n {n}')
        best = max(spare, key=lambda k: (quota[k] - alloc[k], -k))
        alloc[best] += 1
    chosen = []
    for k in keys:
        if alloc[k]:
            pool = strata[k]
            picked = rng.choice(len(pool), size=alloc[k], replace=False)
            chosen.extend(pool[i] for i in picked)
    return sorted(chosen)

def main(argv=None):
    args = parser().parse_args(argv)
    import numpy as np
    from linear_probe.config import PathsConfig, get_task
    from linear_probe.splits import load_public_splits
    from linear_probe.cohort import resolve_sample_paths
    from zeroshot.data.tasks import TASKS

    labels = list(TASKS['csn'].labels)
    split_csv = CSN_SPLIT_ROOT / f'{args.split}.csv'
    positives, row_of = {}, {}
    with split_csv.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            positives[row['sample_id']] = int(row.get('label_count') or 0)
            row_of[row['sample_id']] = row
    # CSN has no report field in the .mat, so SGL alignment uses the whole diagnosis
    # (CSN_index_with_images.csv -> dx_conditions, full condition names, all of them),
    # NOT one text per individual label.
    conditions = {}
    if args.target_text:
        index_csv = Path(args.csn_root) / 'CSN_index_with_images.csv'
        with index_csv.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                conditions[row['sample_id']] = (row.get('dx_conditions') or '').strip()

    def alignment_text(sid):
        raw = conditions.get(sid, '')
        return '. '.join(p.strip() for p in raw.split(';') if p.strip())

    cohort = load_public_splits(get_task('chapman'))
    ids = list(cohort.ids[args.split])
    if args.exclude:
        used = set()
        for path in args.exclude:
            with Path(path).open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    used.add(row['sample_id'])
        before = len(ids)
        ids = [i for i in ids if i not in used]
        print(f'  excluded {before - len(ids)} already-drawn samples '
              f'({len(used)} ids listed in {len(args.exclude)} manifest(s))')
    if args.require_positive:
        ids = [i for i in ids if positives.get(i, 0) > 0]
    if args.n > len(ids):
        raise ValueError(f'--n {args.n} exceeds {len(ids)} eligible {args.split} samples')
    rng = np.random.default_rng(args.seed)
    if args.stratify:
        chosen = stratified_pick(ids, positives, args.n, rng)
    else:
        chosen = sorted(ids[i] for i in rng.choice(len(ids), size=args.n, replace=False).tolist())
    paths = resolve_sample_paths(cohort, PathsConfig.from_env(), check_exists=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'dataset', 'signal_path', 'image_path',
                    'target_class', 'target_text', 'task'])
        for sid in chosen:
            p = paths[sid]
            if p.signal is None or p.image is None:
                raise ValueError(f'{sid} is missing a signal or image path')
            w.writerow([sid, 'csn', str(p.signal), str(p.image), '',
                        alignment_text(sid), 'chapman'])
    counts = [positives[i] for i in chosen]
    print(f'{len(chosen)} CSN {args.split} samples -> {out}')
    print('  ids:', ' '.join(chosen))
    print('  label_count per sample:', counts, '(sum=%d)' % sum(counts))
    if conditions:
        ex = chosen[0]
        print(f'  target_text (report-equivalent) example:')
        print(f'    {ex}: {alignment_text(ex)[:150]}')
    print(f'  (seed={args.seed}, stratify={args.stratify}, eligible={len(ids)}, '
          f'pool mean label_count={sum(positives[i] for i in ids)/len(ids):.2f})')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
