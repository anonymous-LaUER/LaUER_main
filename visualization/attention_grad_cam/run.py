"""python -m visualization.attention_grad_cam.run --help

Default behaviour is a full per-layer export: EVERY attention layer of the targeted
module is rendered for EVERY ground-truth label of the sample. Nothing is pre-selected.

Output layout
    <output>/<stage>/<run_id>/<dataset>_<sample>/<target>/L<kk>/{image,matrix,signal}.png
    <output>/<stage>/<run_id>/<dataset>_<sample>/maps.npz
    <output>/<stage>/<run_id>/<dataset>_<sample>/metadata.json
    <output>/<stage>/<run_id>/summary.json
    <output>/<stage>/<run_id>/index.html

Label space is dataset-dependent and never mixed:
    MEETI  -> the 169-label pretraining bank (data/annotations/)
    CSN    -> CSN's own 38 MELP labels (zeroshot/mappings/csn_melp38.json), expanded
              through the frozen prompt snapshot so MedCPT sees clinical phrases.
              Ground truth comes from zeroshot/splits/csn/<split>.csv.
"""
import argparse
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import re
import traceback

import numpy as np
import torch

from visualization.grad_cam.checkpoint_io import (ROOT, LABEL_ROOT, CSN_SPLIT_ROOT, CPSC_SPLIT_ROOT,
                                                  CSN_PROMPT_PATH, MEETI_FAMILIES, CSN_FAMILIES,
                                                  CPSC_FAMILIES, load_model, load_label_space, read_labels,
                                                  read_ground_truth, read_csn_ground_truth)
from visualization.grad_cam.datasets import Sample, read_manifest, task_samples, load_sample, display_map
from .engine import Engine
from .text import sample_text
from .rendering import render

STAGES = ('sgl', 'sgl_cgl', 'sgl_cgl_tgl')
KINDS = ('image', 'matrix', 'signal')

CLEAN_TOKEN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}')

def slug(value):
    """Folder-safe name. An already-clean ASCII token is used verbatim so the structural
    categories stay readable (signal/, image/, report/, explanation/, SR/, AFIB/); anything
    else gets a truncated readable stem plus a hash so distinct labels stay distinct."""
    text = str(value)
    if CLEAN_TOKEN.fullmatch(text):
        return text
    clean = re.sub(r'[^\w.-]+', '_', text).strip('_.')[:75] or 'sample'
    return clean + '_' + hashlib.sha256(text.encode()).hexdigest()[:8]

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def parser():
    p = argparse.ArgumentParser(description='Three-stage ECG Grad-CAM; never trains a head.')
    p.add_argument('--weights-dir', default=str(ROOT / 'weights/stages'))
    p.add_argument('--output', default=str(Path(__file__).parent / 'output'))
    p.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    p.add_argument('--manifest')
    p.add_argument('--dataset', choices=['meeti', 'ptbxl', 'icbeb', 'cpsc2018', 'chapman', 'csn'],
                   default='meeti')
    p.add_argument('--signal'); p.add_argument('--image'); p.add_argument('--sample-id')
    p.add_argument('--task', help='linear_probe task: ptbxl_super_class/sub_class/form/rhythm, icbeb, chapman')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--ids', nargs='+'); p.add_argument('--limit', type=int)
    p.add_argument('--random', type=int, help='Draw this many samples at random from --ids/--split')
    p.add_argument('--target-class', default='atrial fibrillation')
    p.add_argument('--target-text', default=None,
                   help='Optional override for sample report + explanation alignment text')
    p.add_argument('--fusion-target-class', action='store_true',
                   help='Also run SGL CAM using existing downstream classifier logits')
    p.add_argument('--cgl-target-class', action='store_true',
                   help='Also run a manual CGL target-class CAM, in addition to ground truth')
    p.add_argument('--modalities', nargs='+', choices=['signal', 'image'], default=['signal', 'image'])
    p.add_argument('--encoder-target', choices=['text', 'downstream'], default='text',
                   help='Legacy: downstream also enables fusion-target-class; text CAM is retained')
    p.add_argument('--fusion-target', choices=['native', 'downstream'], default='native')
    p.add_argument('--label-space', choices=['auto', 'meeti', 'csn', 'cpsc2018'], default='auto',
                   help='Which label bank to build the query space from; auto follows --dataset')
    p.add_argument('--prompt-path', default=None,
                   help='Prompt snapshot for CSN labels; '
                        'default = zeroshot/prompts/eval_prompt.json')
    p.add_argument('--csn-split-root', default=str(CSN_SPLIT_ROOT))
    p.add_argument('--cpsc-split-root', default=str(CPSC_SPLIT_ROOT))
    p.add_argument('--skip-fusion', action='store_true'); p.add_argument('--skip-generation', action='store_true')
    p.add_argument('--label-root', default=str(LABEL_ROOT))
    p.add_argument('--head-root', default=str(ROOT / 'output/linear_probe/cache/probe_heads'))
    p.add_argument('--probe-ratio', type=float, default=1.0)
    p.add_argument('--probe-budget', default='melp'); p.add_argument('--probe-seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--amp', choices=['none', 'bf16', 'fp16'], default='none')
    p.add_argument('--seed', type=int, default=42); p.add_argument('--max-generation-len', type=int, default=509)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--token-span', help='Optional absolute generated token range START:END (end exclusive)')
    p.add_argument('--alpha', type=float, default=1.); p.add_argument('--dpi', type=int, default=200)
    p.add_argument('--heat-gamma', type=float, default=.7,
                   help='ALBEF getAttMap blends heat**gamma; 1.0 restores a linear alpha ramp')
    p.add_argument('--cmap', default='jet', help='Matplotlib colormap; blue contrasts with pink ECG paper')
    p.add_argument('--attention-layer', type=int, default=None,
                   help='Single attention layer index (Python, modulo target depth). '
                        'Default None = export EVERY layer')
    p.add_argument('--layers', nargs='+', type=int,
                   help='Explicit subset of layer indices; overrides --attention-layer')
    p.add_argument('--kinds', nargs='+', choices=list(KINDS), default=list(KINDS))
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--pdf', action='store_true'); p.add_argument('--fail-fast', action='store_true')
    return p

def stage_tasks(args, stage):
    return {
        'text': stage == 'sgl',
        'encoder_class': stage == 'sgl' and (
            args.fusion_target_class or args.encoder_target == 'downstream'),
        'fusion_class': stage == 'sgl_cgl' and args.cgl_target_class and not args.skip_fusion,
        'ground_truth': stage == 'sgl_cgl' and not args.skip_fusion,
        'generation': stage == 'sgl_cgl_tgl' and not args.skip_generation,
    }

def layers_for(args, engine, kind):
    """Layer indices to export. Default: every layer of the targeted module."""
    if args.layers:
        return list(args.layers)
    if args.attention_layer is not None:
        return [args.attention_layer]
    m = engine.model
    if kind == 'encoder':
        return list(range(m.ecg_encoder.encoder.depth))
    if kind == 'multilabel':
        return list(range(len(m.query_guided_tracing_head.layers)))
    if kind == 'fusion':
        stack = m.fusion_si_layers if m._use_alternating_fusion else m.fusion_si.layers
        return list(range(len(stack)))
    return list(range(len(m.tgl.layers)))

def select_samples(args):
    if args.manifest:
        samples = read_manifest(args.manifest)
    elif args.task and not (args.signal or args.image):
        samples = task_samples(args.task, args.split, args.ids, None)
    elif args.signal or args.image:
        samples = [Sample(args.sample_id or Path(args.signal or args.image).stem, args.dataset,
                          args.signal or '', args.image or '', task=args.task or '')]
    else:
        raise ValueError('Supply --manifest, --task, or --signal/--image')
    if args.random:
        if args.random > len(samples):
            raise ValueError(f'--random {args.random} exceeds the {len(samples)} available samples')
        order = np.random.default_rng(args.seed).permutation(len(samples))[:args.random]
        samples = [samples[i] for i in sorted(order.tolist())]
    if args.limit:
        samples = samples[:args.limit]
    return samples

def resolve_label_space(args, samples):
    space = args.label_space
    if space == 'auto':
        families = {str(s.dataset).lower() for s in samples}
        for key, group in (('meeti', MEETI_FAMILIES), ('csn', CSN_FAMILIES),
                           ('cpsc2018', CPSC_FAMILIES)):
            if families <= group:
                space = key
                break
        else:
            raise ValueError(f'--label-space auto cannot mix families: {sorted(families)}')
    names, queries = load_label_space(space, args.label_root, args.prompt_path)
    if space == 'meeti':
        truth = read_ground_truth(args.label_root, samples, queries)
    else:
        root = args.csn_split_root if space == 'csn' else args.cpsc_split_root
        truth = read_csn_ground_truth(samples, names, queries, root)
    return space, names, queries, truth

def main(argv=None):
    args = parser().parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    if (not 0 <= args.alpha <= 1 or not 0 < args.heat_gamma <= 1
            or args.max_generation_len < 1 or args.temperature <= 0):
        raise ValueError('Invalid alpha/heat-gamma/generation parameters')
    samples = select_samples(args)
    space, names, labels, ground_truth = resolve_label_space(args, samples)
    print(f'[CAM] label space = {space} ({len(labels)} queries); {len(samples)} samples', flush=True)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = Path(args.output).resolve()
    index_dir = output / 'comparisons' / run_id
    index_dir.mkdir(parents=True, exist_ok=True)
    dump(index_dir / 'arguments.json', vars(args))
    dump(index_dir / 'label_space.json',
         {'space': space, 'n_queries': len(labels), 'names': names, 'queries': labels,
          'prompt_path': (str(args.prompt_path) if args.prompt_path
                          else (str(CSN_PROMPT_PATH) if space == 'csn' else None))})
    records = []
    for stage in args.stages:
        tasks = stage_tasks(args, stage)
        if not any(tasks.values()):
            print(f'[CAM] {stage}: skipped (no enabled tasks)', flush=True)
            continue
        stage_dir = output / stage / run_id
        stage_dir.mkdir(parents=True, exist_ok=True)
        dump(stage_dir / 'arguments.json', vars(args))
        model, model_meta = load_model(Path(args.weights_dir) / (stage + '.pt'), args.device, labels, stage_dir)
        if model_meta['source_mode'].replace('+', '_') != stage:
            raise ValueError('Checkpoint stage does not match filename')
        meta_model = dict(model_meta, label_space=space, label_space_names=names)
        dump(stage_dir / 'model.json', meta_model)
        engine = Engine(model, args.device, args.amp)
        heads = {}

        def get_head(sample, mode):
            task = sample.task or args.task
            if not task:
                raise ValueError('Downstream CAM requires --task or task column')
            key = (task, mode)
            if key not in heads:
                head, payload, path = reuse_head(args.head_root, model_meta['sha256'], task, mode,
                    args.probe_ratio, args.probe_budget, args.probe_seed, model)
                heads[key] = head.to(args.device), payload, path
            head, payload, path = heads[key]
            target = sample.target_class or args.target_class
            hnames = payload['identity']['label_names']
            if target not in hnames:
                raise ValueError(f'Downstream target {target!r} not in {hnames}')
            amp = payload['identity']['encoder_tag'].rsplit('-amp', 1)[1]
            if args.device == 'cpu' and amp != 'none':
                raise ValueError('This saved head used CUDA AMP; use CUDA for faithful replay')
            return head, hnames.index(target), amp, path

        stage_records = []
        for sample in samples:
            folder = stage_dir / slug(sample.dataset) / slug(sample.sample_id)
            folder.mkdir(parents=True, exist_ok=True)
            meta = {'sample': vars(sample), 'stage': stage, 'label_space': space,
                    'layers': [], 'maps': [], 'errors': []}
            arrays = {}

            def save(name, target_label, layer, result, kinds, body='', slug_label=None):
                maps, info = result
                tslug = slug(str(slug_label if slug_label is not None else target_label))
                ldir = folder / tslug / f'L{layer:02d}'
                ldir.mkdir(parents=True, exist_ok=True)
                info.update(name=name, target=target_label, target_slug=tslug, layer_index=layer,
                            display_leads=list(data['leads']), native_grid=[12, 40], patch_seconds=.25,
                            heat_gamma=args.heat_gamma, alpha=args.alpha)
                meta['maps'].append(info)
                if layer not in meta['layers']:
                    meta['layers'].append(layer)
                for k, v in maps.items():
                    arrays[f'L{layer:02d}__{tslug}__{k}'] = v
                heat = display_map(maps['normalized'], data)
                subtitle = (f"layer {layer} | {info['layer']} | {info['score_type']}={info['score']:.5f}"
                            f" | {info['status']} | eff_cells={info['effective_cells']:.0f}/480"
                            f" | top8={info['top8_mass']:.0%}")
                if 'probability' in info:
                    subtitle += f" | sigmoid={info['probability']:.4f}"
                if info.get('saturated'):
                    cell = (f"{info['effective_cells']:.0f}/480 effective cells, "
                            f"top-8 = {info['top8_mass']:.0%} of mass")
                    subtitle += f" | WARNING saturated: {cell}"
                    meta.setdefault('warnings', []).append(
                        {'map': name, 'layer': layer, 'kind': 'saturated_attention', 'detail': cell})
                    print(f'[CAM]   warning {sample.sample_id}/{name} L{layer}: saturated ({cell})', flush=True)
                title = f'{stage} | {sample.sample_id} | {target_label}'
                # matrix is always rendered: the final folder is image + matrix + signal.
                for kind in dict.fromkeys(list(kinds) + ['matrix']):
                    render(ldir / f'{kind}.png', data, heat, title, subtitle, kind, body,
                           args.alpha, args.dpi, args.pdf, args.cmap, gamma=args.heat_gamma)

            def sweep(name, target_label, kind_of_layers, make, kinds, body='', slug_label=None,
                      layers=None):
                for k in (layers if layers is not None
                          else layers_for(args, engine, kind_of_layers)):
                    engine.attention_layer = k
                    save(name, target_label, k, make(), kinds, body, slug_label)

            def attempt(name, fn):
                try:
                    fn()
                except Exception as exc:
                    meta['errors'].append({'task': name, 'error': str(exc), 'traceback': traceback.format_exc()})
                    if args.fail_fast:
                        raise

            try:
                fusion = tasks['fusion_class'] or tasks['ground_truth']
                generation = tasks['generation']
                data = load_sample(sample, args.device,
                    'signal' in args.modalities or fusion or generation,
                    'image' in args.modalities or fusion or generation)
                target = sample.target_class or args.target_class
                text = (sample_text(sample, model.config.data.text_separator, args.target_text)
                        if tasks['text'] else '')
                if tasks['text']:
                    meta['alignment_text'] = text
                # SGL renders the last three encoder blocks (not one, not all twelve).
                # --layers / --attention-layer still override this when given explicitly.
                depth = model.ecg_encoder.encoder.depth
                fusion_layers = (layers_for(args, engine, 'encoder')
                                 if (args.layers or args.attention_layer is not None)
                                 else list(range(max(depth - 3, 0), depth)))
                for modality in args.modalities if tasks['text'] else []:
                    # slug_label=m keeps signal/ and image/ apart; without it both
                    # modalities of SGL would write into the same target folder.
                    # Both modalities render image + matrix + signal so every leaf folder
                    # holds the same three views as Stages 2 and 3.
                    attempt('encoder_' + modality, lambda m=modality: sweep(
                        'encoder_' + m, 'Long-text alignment', 'encoder',
                        lambda: engine.encoder_cam(data, m, text), ['image', 'signal'], text,
                        slug_label=m, layers=fusion_layers))
                for modality in args.modalities if tasks['encoder_class'] else []:
                    def encoder_class_job(m=modality):
                        head, idx, amp, path = get_head(sample, m + '_only')
                        meta.setdefault('heads', {})[m] = path
                        sweep('encoder_classification_' + m, target, 'encoder',
                              lambda: engine.encoder_cam(data, m, text, head, idx, amp), [m])
                    attempt('encoder_classification_' + modality, encoder_class_job)
                if tasks['ground_truth']:
                    def ground_truth_job():
                        truth = ground_truth.get(sample.sample_id)
                        if truth is None:
                            raise ValueError(f'No {space} ground truth for {sample.dataset}/{sample.sample_id}')
                        meta['ground_truth'] = truth
                        if not truth['labels']:
                            meta.setdefault('skipped', []).append(
                                {'task': 'ground_truth', 'reason': 'no positive labels'})
                        shorts = truth.get('short_labels') or truth['labels']
                        for label, short in zip(truth['labels'], shorts):
                            sweep('fusion_ground_truth', label, 'multilabel',
                                  lambda lb=label: engine.classification_cam(data, labels.index(lb)),
                                  ['image', 'signal'], slug_label=short)
                    attempt('ground_truth', ground_truth_job)
                if tasks['fusion_class']:
                    def classification_job():
                        head = None; amp = None
                        if args.fusion_target == 'downstream':
                            head, idx, amp, path = get_head(sample, 'fusion')
                            meta.setdefault('heads', {})['fusion'] = path
                            kind_of_layers = 'fusion'
                        else:
                            if target not in labels:
                                raise ValueError(f'Native target must be an exact label: {target}')
                            idx = labels.index(target)
                            kind_of_layers = 'multilabel'
                        sweep('fusion_classification', target, kind_of_layers,
                              lambda: engine.classification_cam(data, idx, head, amp), ['image', 'signal'])
                    attempt('fusion_classification', classification_job)
                if generation:
                    def generation_job():
                        ids, regions, generated = engine.generate(data, args.seed, args.max_generation_len,
                            args.temperature, sample.dataset + '/' + sample.sample_id)
                        dump(folder / 'generated_text.json', generated)
                        if args.token_span:
                            a, b = map(int, args.token_span.split(':'))
                            valid = set(regions['report'] + regions['explanation'])
                            if not set(range(a, b)).issubset(valid) or b <= a:
                                raise ValueError('Token span must select generated content only')
                            regions['selected_span'] = list(range(a, b))
                        for region, positions in regions.items():
                            if not positions:
                                meta.setdefault('skipped', []).append(
                                    {'task': region, 'reason': 'no generated content'})
                                continue
                            body = model.text_encoder.tokenizer.decode(ids[0, positions], skip_special_tokens=True)
                            sweep('fusion_generation', region, 'tgl',
                                  lambda p=positions: engine.generation_cam(data, ids, p),
                                  ['image', 'signal'], body)
                    attempt('generation', generation_job)
            except Exception as exc:
                meta['errors'].append({'task': 'sample', 'error': str(exc), 'traceback': traceback.format_exc()})
                if args.fail_fast:
                    dump(folder / 'metadata.json', meta)
                    raise
            np.savez_compressed(folder / 'maps.npz', **arrays)
            dump(folder / 'metadata.json', meta)
            # Per-stage list: summarise() must never see another stage's maps.
            stage_records.append({'stage': stage, 'dataset': sample.dataset, 'sample_id': sample.sample_id,
                                  'folder': str(folder), 'maps': meta['maps'],
                                  'map_count': len(meta['maps']),
                                  'layers': sorted(meta['layers']), 'errors': meta['errors']})
            print(f"[CAM] {stage}/{sample.sample_id}: {len(meta['maps'])} maps over "
                  f"{len(meta['layers'])} layers, {len(meta['errors'])} errors", flush=True)
        summarise(stage_dir, stage, stage_records, space)
        records.extend(stage_records)
        del engine, model, heads
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    dump(index_dir / 'index.json', records)
    print(f'Results: {output}  (run {run_id})', flush=True)
    return 1 if any(r['errors'] for r in records) else 0

def summarise(stage_dir, stage, records, space):
    flat = [dict(m, sample=r['sample_id']) for r in records for m in r['maps']]
    per_layer = {}
    for m in flat:
        per_layer.setdefault(m['layer_index'], []).append(m)
    summary = {'stage': stage, 'label_space': space, 'per_layer': {}, 'per_map': flat,
               'errors': [dict(e, sample=r['sample_id']) for r in records for e in r['errors']]}
    for k in sorted(per_layer):
        rows = per_layer[k]
        summary['per_layer'][f'L{k:02d}'] = {
            'layer': rows[0]['layer'], 'maps': len(rows),
            'mean_effective_cells': round(float(np.mean([m['effective_cells'] for m in rows])), 1),
            'mean_top8_mass': round(float(np.mean([m['top8_mass'] for m in rows])), 4),
            'saturated_maps': int(sum(bool(m['saturated']) for m in rows))}
    dump(stage_dir / 'summary.json', summary)
    html = ['<!doctype html><meta charset="utf-8">', f'<title>{stage} per-layer CAM</title>',
            '<style>body{font-family:system-ui,sans-serif;margin:24px}'
            'img{width:480px;border:1px solid #ccc;margin:4px}'
            'td,th{border-bottom:1px solid #eee;padding:4px 10px;font-size:13px;text-align:left}'
            'h2{margin-top:32px}</style>',
            f'<h1>{stage} - per-layer Grad-CAM</h1><p>label space: {space}</p>',
            '<table><tr><th>layer</th><th>hook</th><th>maps</th><th>mean eff cells /480</th>'
            '<th>mean top-8</th><th>saturated</th></tr>']
    for key, v in summary['per_layer'].items():
        html.append(f"<tr><td>{key}</td><td>{v['layer']}</td><td>{v['maps']}</td>"
                    f"<td>{v['mean_effective_cells']}</td><td>{v['mean_top8_mass']:.3f}</td>"
                    f"<td>{v['saturated_maps']}</td></tr>")
    html.append('</table>')
    for k in sorted(per_layer):
        html.append(f"<h2>layer {k} - {per_layer[k][0]['layer']}</h2>")
        for m in per_layer[k]:
            folder = next(r['folder'] for r in records if r['sample_id'] == m['sample'])
            rel = Path(folder).relative_to(stage_dir) / m['target_slug'] / f'L{k:02d}'
            for kind in ('signal', 'matrix'):
                if (stage_dir / rel / f'{kind}.png').exists():
                    html.append(f'<figure style="display:inline-block;margin:6px">'
                                f'<img src="{(rel / f"{kind}.png").as_posix()}" alt="{m["sample"]} {m["target"]}">'
                                f'<figcaption style="font-size:12px">{m["sample"]} | {m["target"]} | '
                                f'{kind} | eff={m["effective_cells"]:.0f}</figcaption></figure>')
    (stage_dir / 'index.html').write_text('\n'.join(html), encoding='utf-8')
    print(f'[CAM] {stage}: summary -> {stage_dir / "summary.json"}', flush=True)
    for key, v in summary['per_layer'].items():
        print(f'[CAM]   {key} {v["layer"]}: mean eff={v["mean_effective_cells"]}/480 '
              f'top8={v["mean_top8_mass"]:.3f} over {v["maps"]} maps, {v["saturated_maps"]} saturated',
              flush=True)

if __name__ == '__main__':
    raise SystemExit(main())
