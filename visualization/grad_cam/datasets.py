import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.lead_order import MIMIC_LEAD_ORDER, STANDARD_LEAD_ORDER, signal_to_mimic_order
from linear_probe.config import PathsConfig, PreprocessConfig, get_task
from linear_probe.dataset import read_raw_signal, preprocess_signal, load_preprocessed_image

ALIASES = {'cpsc2018': 'icbeb', 'csn': 'chapman'}

@dataclass
class Sample:
    sample_id: str
    dataset: str
    signal_path: str = ''
    image_path: str = ''
    target_class: str = ''
    target_text: str = ''
    task: str = ''

def read_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    samples = []
    for row in rows:
        values = {k: row.get(k, '') or '' for k in Sample.__dataclass_fields__}
        if not values['sample_id'] or not values['dataset']:
            raise ValueError('Each manifest row requires sample_id and dataset')
        for key in ('signal_path', 'image_path'):
            if values[key]:
                p = Path(values[key])
                values[key] = str(p if p.is_absolute() else path.parent / p)
        samples.append(Sample(**values))
    if not samples:
        raise ValueError('Empty manifest')
    keys = [(s.dataset, s.sample_id) for s in samples]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate dataset/sample IDs in manifest')
    return samples

def task_samples(task, split, ids=None, limit=None):
    from linear_probe.splits import load_public_splits
    from linear_probe.cohort import resolve_sample_paths
    cohort = load_public_splits(get_task(task))
    selected = list(ids) if ids else list(cohort.ids[split])
    if not set(selected).issubset(cohort.ids[split]):
        raise ValueError('Requested IDs are not in the selected split')
    if limit:
        selected = selected[:limit]
    paths = resolve_sample_paths(cohort, PathsConfig.from_env(), check_exists=False)
    return [Sample(i, cohort.family, str(paths[i].signal or ''),
                   str(paths[i].image or ''), task=task) for i in selected]

def load_sample(sample, device, need_signal=True, need_image=True):
    family = ALIASES.get(sample.dataset.lower(), sample.dataset.lower())
    if family not in ('meeti', 'ptbxl', 'icbeb', 'chapman'):
        raise ValueError(f'Unsupported dataset: {family}')
    native = family == 'meeti'
    data = {'leads': MIMIC_LEAD_ORDER if native else STANDARD_LEAD_ORDER,
            'swap': not native, 'signal': None, 'image': None,
            'waveform': None, 'original_image': None, 'valid_samples': 5000}
    if need_signal:
        if not sample.signal_path:
            raise ValueError('Signal path required')
        if native:
            # Mirror MEETI's signal-field semantics but fail instead of its
            # DDP-specific zero fallback (a fabricated CAM would be misleading).
            from scipy.io import loadmat
            raw = loadmat(sample.signal_path, variable_names=['signal'])['signal'].astype(np.float32)
            if raw.shape != (12, 5000):
                raise ValueError(f'MEETI signal shape must be (12,5000), got {raw.shape}')
            raw = np.nan_to_num(raw, nan=0, posinf=0, neginf=0)
            tensor = torch.from_numpy(raw)[None]
        else:
            if family in ('ptbxl', 'chapman'):
                import wfdb
                base = str(Path(sample.signal_path).with_suffix(''))
                header = wfdb.rdheader(base)
                if header.fs != 500:
                    raise ValueError(f'Expected native 500 Hz record, got {header.fs}; no silent resampling')
                observed = [n.upper() for n in header.sig_name]
                if observed != [n.upper() for n in STANDARD_LEAD_ORDER]:
                    raise ValueError(f'Unexpected native lead order: {header.sig_name}')
            raw = read_raw_signal(sample.signal_path, family, PreprocessConfig())
            tensor = torch.from_numpy(preprocess_signal(raw, PreprocessConfig()))[None]
        data['valid_samples'] = min(raw.shape[-1], 5000)
        wave = tensor[0].numpy()
        data['waveform'] = signal_to_mimic_order(wave) if not native else wave.copy()
        data['signal'] = tensor.to(device)
    if need_image:
        if not sample.image_path:
            raise ValueError('Image path required; zero-image fallback is disabled')
        with Image.open(sample.image_path) as im:
            original = np.array(im.convert('RGB'))
        if original.shape != (720, 1000, 3):
            raise ValueError(f'Expected original 12-row 720x1000 image, got {original.shape}')
        data['original_image'] = original
        image = (original.transpose(2, 0, 1).copy() if native else
                 load_preprocessed_image(sample.image_path, PreprocessConfig()))
        data['image'] = torch.from_numpy(image)[None].to(device)
    return data

def display_map(cam, data):
    return signal_to_mimic_order(cam) if data['swap'] else cam.copy()
