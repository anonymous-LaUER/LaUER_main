import csv
import hashlib
import json
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
LABEL_ROOT = ROOT / 'data/annotations'
# CSN (chapman) labels live with the zero-shot benchmark, NOT in the MEETI pretraining CSVs.
CSN_SPLIT_ROOT = ROOT / 'zeroshot/splits/csn'
CPSC_SPLIT_ROOT = ROOT / 'zeroshot/splits/cpsc2018'
# Frozen query snapshot used to expand MELP acronyms (CSN 38 / CPSC 9) into clinical phrases.
CSN_PROMPT_PATH = ROOT / 'zeroshot/prompts/official_CSN.json'
CPSC_PROMPT_PATH = ROOT / 'zeroshot/prompts/official_CPSC.json'
MEETI_FAMILIES = frozenset({'meeti'})
CSN_FAMILIES = frozenset({'csn', 'chapman'})
CPSC_FAMILIES = frozenset({'cpsc2018', 'icbeb'})
# Datasets whose labels live with the zero-shot benchmark, not in the MEETI CSVs.
MELP_FAMILIES = CSN_FAMILIES | CPSC_FAMILIES
MELP_TASK_KEY = {'csn': 'csn', 'chapman': 'csn', 'cpsc2018': 'cpsc2018', 'icbeb': 'cpsc2018'}
MELP_SPLIT_ROOT = {'csn': CSN_SPLIT_ROOT, 'chapman': CSN_SPLIT_ROOT,
                   'cpsc2018': CPSC_SPLIT_ROOT, 'icbeb': CPSC_SPLIT_ROOT}

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_labels(root=LABEL_ROOT):
    headers = []
    for split in ('train', 'val', 'test'):
        with open(Path(root) / f'{split}.csv',
                  encoding='utf-8-sig', newline='') as f:
            headers.append(next(csv.reader(f)))
    if not all(h == headers[0] for h in headers):
        raise ValueError('Pretraining split label orders differ')
    if headers[0][:3] != ['subject_id', 'study_id', 'total_report']:
        raise ValueError('Unexpected pretraining CSV schema')
    labels = headers[0][3:]
    if len(labels) != len(set(labels)):
        raise ValueError('Duplicate native label names')
    return labels

def read_ground_truth(root, samples, labels):
    """Read requested MEETI study IDs only; keep native classifier column order."""
    wanted = {s.sample_id for s in samples if s.dataset.lower() == 'meeti'}
    found = {}
    if not wanted:
        return found
    for split in ('train', 'val', 'test'):
        path = Path(root) / f'{split}.csv'
        with path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            if header[3:] != labels:
                raise ValueError(f'Ground-truth label order mismatch: {path}')
            id_col = header.index('study_id')
            for row in reader:
                sid = row[id_col]
                if sid not in wanted:
                    continue
                if sid in found:
                    raise ValueError(f'Duplicate ground-truth study_id: {sid}')
                values = [float(v) for v in row[3:]]
                if len(values) != len(labels) or any(v not in (0., 1.) for v in values):
                    raise ValueError(f'Invalid binary ground truth: {sid}')
                found[sid] = {'source': str(path.resolve()), 'study_id': sid,
                              'labels': [name for name, v in zip(labels, values) if v == 1.]}
    return found

def read_melp_label_names(task_key):
    """Task labels from zeroshot/data/tasks.py (CSN 38, CPSC 9). Not the MEETI bank."""
    from zeroshot.data.tasks import TASKS
    return list(TASKS[task_key].labels)

def read_csn_label_names():
    """CSN's own 38 MELP labels."""
    return read_melp_label_names('csn')

def load_label_space(dataset, label_root=LABEL_ROOT, prompt_path=None):
    """Return (names, queries) for the dataset family.

    names   -- human label identifiers, used to look up ground truth
    queries -- the exact strings the multi-label query bank is built from

    MEETI uses the 169-label pretraining bank verbatim. CSN uses its own 38 labels,
    expanded through the frozen MELP prompt snapshot (zeroshot/prompts), because the
    raw acronyms (SR/WPW/2AVB) carry no meaning for MedCPT.
    """
    family = str(dataset).lower()
    if family in MELP_FAMILIES:
        names = read_melp_label_names(MELP_TASK_KEY[family])
        from zeroshot.prompts import prompts_for_labels
        # Raw acronyms (SR/WPW/2AVB, NORM/CRBBB) mean nothing to MedCPT, so the query bank is
        # always built from a frozen prompt snapshot unless the caller overrides it.
        # Default is eval_prompt.json (not melp_ckepe.json).
        default = CSN_PROMPT_PATH if family in CSN_FAMILIES else CPSC_PROMPT_PATH
        queries = list(prompts_for_labels(names, prompt_path or default, task=family))
        return names, queries
    labels = read_labels(label_root)
    return labels, labels

def read_csn_ground_truth(samples, names, queries, split_root=CSN_SPLIT_ROOT):
    """Per-sample MELP positives from zeroshot/splits/<task>/<split>.csv.

    Label columns are 0/1; positives are returned in the query space so the caller can
    index the same bank the model encodes. Handles CSN (38 labels) and CPSC (9 labels).
    """
    wanted = {s.sample_id for s in samples if str(s.dataset).lower() in MELP_FAMILIES}
    found = {}
    if not wanted:
        return found
    root = Path(split_root)
    for split in ('train', 'val', 'test'):
        path = root / f'{split}.csv'
        if not path.is_file():
            continue
        with path.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                sid = row.get('sample_id')
                if sid not in wanted:
                    continue
                if sid in found:
                    raise ValueError(f'Duplicate CSN ground-truth sample_id: {sid}')
                values = [int(float(row.get(name, 0) or 0)) for name in names]
                found[sid] = {'source': str(path.resolve()), 'study_id': sid, 'split': split,
                              'diagnose': row.get('diagnose', ''),
                              # labels = query strings (what the model encodes);
                              # short_labels = the CSN acronyms, used for folder names.
                              'labels': [q for n, q, v in zip(names, queries, values) if v == 1],
                              'short_labels': [n for n, v in zip(names, values) if v == 1]}
    return found

def load_model(path, device, labels, audit_dir):
    from configs.config import load_config_from_yaml
    from models.lauer import LaUER
    from training.stage_io import artifact_source_mode, extract_model_state
    artifact = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    state = extract_model_state(artifact)
    mode = artifact_source_mode(artifact, state)
    config = json.loads(json.dumps(artifact['config']))
    # Only local runtime paths are relocated. Architecture comes from the artifact.
    article = str(ROOT / 'weights/MedCPT-Article-Encoder')
    query = str(ROOT / 'weights/MedCPT-Query-Encoder')
    config['text_encoder'].update(model_name_or_path=article,
        tokenizer_name_or_path=article, query_model_name_or_path=query)
    config['tgl']['medcpt_path'] = article
    config['ecg_encoder']['vit1d']['checkpoint_path'] = ''
    config['image_encoder']['vit2d']['checkpoint_path'] = ''
    config['train']['nan_debug'] = False
    audit_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = audit_dir / 'resolved_config.yaml'
    cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    model = LaUER(load_config_from_yaml(str(cfg_path)), mode=mode)
    model.load_pretrained_encoders(device='cpu')
    state = {k.removeprefix('module.'): v for k, v in state.items()
             if 'text_adapter.proj1_cgl.' not in k}
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith(('text_encoder.', 'query_encoder.'))]
    if bad or unexpected:
        raise ValueError(f'Checkpoint mismatch: missing={bad}, unexpected={unexpected}')
    model.to(device).eval().requires_grad_(False)
    if model.has_cgl:
        model.prepare_prompt_bank(labels, device=device)
    return model, {'checkpoint': str(Path(path).resolve()), 'sha256': digest(path),
                   'source_mode': mode, 'label_source': 'user-specified annotation CSVs',
                   'labels': labels, 'resolved_config': str(cfg_path)}

def reuse_head(cache_root, checkpoint_sha, task, mode, ratio, budget, seed, model=None):
    """Reuse exact BCE head; validate full cohort/preprocess identity, never train."""
    from linear_probe.config import PreprocessConfig, get_task
    from linear_probe.splits import load_public_splits
    from linear_probe.head_io import head_identity, load_head
    matches = []
    for path in Path(cache_root).rglob('*.pt'):
        if f'-{checkpoint_sha[:16]}-' not in path.parent.name:
            continue
        a = torch.load(path, map_location='cpu', weights_only=True)
        identity = a.get('identity', {})
        if all(identity.get(k) == v for k, v in dict(task=task, mode=mode,
                ratio=ratio, budget=budget, seed=seed).items()):
            matches.append((path, identity))
    if len(matches) != 1:
        raise ValueError(f'Expected one matching trained head, found {len(matches)}: {task}/{mode}')
    path, identity = matches[0]
    # Reproduce the YAML that generated cached probe features; filenames alone
    # cannot establish that the fusion routing/pooling matches.
    config_hash = identity['encoder_tag'].split('-cfg', 1)[1].split('-amp', 1)[0]
    configs = [p for p in (ROOT / 'configs').rglob('*.yaml')
               if digest(p).startswith(config_hash)]
    if not configs:
        raise ValueError('Original probe config fingerprint not found in configs/')
    if model is not None:
        from configs.config import load_config_from_yaml
        from training.stage_io import architecture_signature
        probe_config = load_config_from_yaml(str(configs[0]))
        if architecture_signature(probe_config, model.mode) != architecture_signature(model.config, model.mode):
            raise ValueError('Probe feature architecture differs from checkpoint runtime')
    if identity['feature_kind'] not in ('cls', 'fusion_cls'):
        raise ValueError('Only existing raw CLS/fusion CLS heads supported')
    cohort = load_public_splits(get_task(task))
    expected = head_identity(identity['encoder_tag'], PreprocessConfig(), cohort,
                             task, mode, 'cls', ratio, budget, seed)
    head, payload = load_head(path, expected)
    return head, payload, str(path)
