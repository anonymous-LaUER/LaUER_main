"""MEETI signal-image-text loading and split-specific clinical annotations.

Signals have shape [12, 5000], images [3, 720, 1000]. SGL text concatenates
report [SEP] explanation. TGL text adds literal report:/explanation: markers
and a terminating separator. Label columns follow the annotation CSV order."""
import os
import csv
import logging
from typing import Optional, List, Dict

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from scipy.io import loadmat

from training.reproducibility import make_data_generator, seed_worker

_SAMPLE_FIELDS = ('sample_id', 'mat_path', 'png_path', 'has_image')

class _Sample(tuple):
    __slots__ = ()

    def __new__(cls, sample_id, mat_path, png_path, has_image=True):
        return tuple.__new__(cls, (sample_id, mat_path, png_path, has_image))

    def __getitem__(self, key):
        if isinstance(key, str):
            return tuple.__getitem__(self, _SAMPLE_FIELDS.index(key))
        return tuple.__getitem__(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except (ValueError, IndexError):
            return default

    @property
    def sample_id(self):
        return tuple.__getitem__(self, 0)

    @property
    def mat_path(self):
        return tuple.__getitem__(self, 1)

    @property
    def png_path(self):
        return tuple.__getitem__(self, 2)

    @property
    def has_image(self):
        return tuple.__getitem__(self, 3)

def _evict_file_pages(path: str) -> None:
    if os.name != 'posix':
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass

logger = logging.getLogger(__name__)

def _build_text(report, llm_text, separator):
    return f"{report}{separator}{llm_text}"

def _hash_split(sample_id: str) -> str:
    import hashlib
    h = int(hashlib.md5(sample_id.encode()).hexdigest(), 16) % 10
    if h < 8:
        return 'train'
    elif h == 8:
        return 'val'
    else:
        return 'test'

def _norm_data_root(data_root):
    if not data_root:
        return data_root
    return data_root.strip().replace("'", "").replace('"', "")

def _resolve_index_dir(data_root: str, index_dir: str = None) -> str:
    if index_dir:
        return index_dir
    data_root = _norm_data_root(data_root)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_idx = os.path.join(repo_root, 'file_index')
    if (os.path.isdir(local_idx)
            and os.path.exists(os.path.join(local_idx, 'train.txt'))):
        return local_idx
    return os.path.join(data_root, 'file_index')

def _load_split_from_index(data_root: str, index_dir: str, split: str):
    data_root = _norm_data_root(data_root)
    idx_file = os.path.join(index_dir, f'{split}.txt')
    if not os.path.exists(idx_file):
        return None

    samples = []
    with open(idx_file, 'r') as f:
        for line in f:
            rel_path = line.strip()
            if not rel_path:
                continue
            abs_mat = os.path.join(data_root, rel_path)
            sid = os.path.splitext(os.path.basename(rel_path))[0]
            abs_png = abs_mat.replace('.mat', '.png')

            samples.append(_Sample(sid, abs_mat, abs_png, True))
    return samples

def _walk_meeti_samples(data_root: str):
    data_root = _norm_data_root(data_root)
    try:
        for group_entry in os.scandir(data_root):
            if not group_entry.is_dir() or not group_entry.name.startswith('p'):
                continue
            for subj_entry in os.scandir(group_entry.path):
                if not subj_entry.is_dir():
                    continue
                for study_entry in os.scandir(subj_entry.path):
                    if not study_entry.is_dir():
                        continue
                    mat_file = None
                    png_file = None
                    for fentry in os.scandir(study_entry.path):
                        if fentry.name.endswith('.mat') and mat_file is None:
                            mat_file = fentry.path
                            sid = fentry.name[:-4]
                        elif fentry.name.endswith('.png') and png_file is None:
                            png_file = fentry.path
                    if mat_file is not None:
                        yield sid, mat_file, png_file
    except FileNotFoundError:
        return

def _scan_all_items(data_root: str, require_image: bool = True):
    global _SCAN_CACHE

    cache_key = (_norm_data_root(data_root), require_image)
    if _SCAN_CACHE is not None and _SCAN_CACHE.get('__key__') == cache_key:
        return _SCAN_CACHE['all']

    items = []
    for sid, mat_path, png_path in _walk_meeti_samples(data_root):
        has_img = png_path is not None and os.path.exists(png_path)
        if require_image and not has_img:
            continue
        items.append(_Sample(sid, mat_path, png_path, has_img))

    _SCAN_CACHE = {'all': items, '__key__': cache_key}
    return items

_SCAN_CACHE = None

def _scan_and_hash_split(data_root: str, split: str, require_image: bool = True):
    return [it for it in _scan_all_items(data_root, require_image)
            if _hash_split(it['sample_id']) == split]

_CSV_STUDY_IDS_CACHE: Dict[str, frozenset] = {}

def _load_csv_study_ids(annotation_csv: str):
    if not annotation_csv:
        return None
    if annotation_csv in _CSV_STUDY_IDS_CACHE:
        return _CSV_STUDY_IDS_CACHE[annotation_csv]
    ids = set()
    with open(annotation_csv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # header: subject_id, study_id, total_report, <labels...>
        for row in reader:
            if len(row) < 2:
                continue
            ids.add(row[1].strip())
    if not ids:
        raise ValueError(f"No study_ids found in {annotation_csv}")
    ids = frozenset(ids)
    _CSV_STUDY_IDS_CACHE[annotation_csv] = ids
    return ids

def _scan_and_csv_split(data_root: str, split: str, require_image: bool = True,
                        split_csv: str = None):
    split_ids = _load_csv_study_ids(split_csv)
    if split_ids is None:
        return _scan_and_hash_split(data_root, split, require_image)
    return [it for it in _scan_all_items(data_root, require_image)
            if it['sample_id'] in split_ids]

ANNOTATION_METADATA_COLUMNS = ("subject_id", "study_id", "total_report")

def read_label_names(annotation_csv: str) -> List[str]:
    """Read concept labels in CSV column order, excluding metadata fields."""
    with open(annotation_csv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
    if tuple(header[:3]) != ANNOTATION_METADATA_COLUMNS:
        raise ValueError(
            f"Expected annotation columns {ANNOTATION_METADATA_COLUMNS}, "
            f"got {tuple(header[:3])}"
        )
    # Target columns and cached query features must retain exactly the same ordering.
    labels = [name.strip() for name in header[3:]]
    if not labels or any(not label for label in labels):
        raise ValueError("Annotation CSV contains empty label names")
    if len(labels) != len(set(labels)):
        raise ValueError("Annotation CSV contains duplicate label column names")

    try:
        from data.label_mapping import load_canonical_labels
        canonical = load_canonical_labels(annotation_csv)
        if canonical is not None and set(labels) != set(canonical):
            raise ValueError(
                f"Annotation CSV label set ({len(labels)}) does not match "
                f"the version-matched canonical label table ({len(canonical)}); "
                f"order may differ, "
                f"set must be equal"
            )
    except ImportError:
        pass
    return labels

def load_annotations(annotation_csv: str, required_study_ids) -> Dict[str, torch.Tensor]:
    required = set(map(str, required_study_ids))
    retained: Dict[str, torch.Tensor] = {}
    with open(annotation_csv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) != len(header):
                continue
            study_id = row[1].strip()
            if study_id not in required:
                continue
            try:
                labels = torch.tensor(
                    [float(value or 0.0) for value in row[3:]], dtype=torch.float32
                )
            except ValueError as error:
                raise ValueError(f"Invalid label value for study {study_id}") from error
            if not bool(((labels == 0) | (labels == 1)).all()):
                raise ValueError(f"Labels must be binary for study {study_id}")
            if study_id in retained:
                raise ValueError(f"Duplicate annotation row for study {study_id}")
            retained[study_id] = labels
    return retained

class _AnnotationTable:

    def __init__(self, study_ids, matrix):
        self._ids = list(study_ids)
        self._matrix = matrix  # [N, C] float32
        self._index = {sid: i for i, sid in enumerate(self._ids)}

    def __contains__(self, study_id):
        return study_id in self._index

    def __len__(self):
        return len(self._ids)

    def get(self, study_id, default=None):
        row = self._index.get(study_id)
        return self._matrix[row] if row is not None else default

def _load_annotations_all(annotation_csv, required_study_ids=None):
    """Load annotation targets once and retain only records needed by the requested split."""
    label_names = read_label_names(annotation_csv)
    required = set(map(str, required_study_ids)) if required_study_ids is not None else None
    study_ids: List[str] = []
    rows: List[list] = []
    seen: set = set()
    with open(annotation_csv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) != len(header):
                continue
            study_id = row[1].strip()
            if required is not None and study_id not in required:
                continue
            if study_id in seen:
                raise ValueError(f"Duplicate annotation row for study {study_id}")
            try:
                values = [float(value or 0.0) for value in row[3:]]
            except ValueError as error:
                raise ValueError(f"Invalid label value for study {study_id}") from error
            if not all(v in (0.0, 1.0) for v in values):
                raise ValueError(f"Labels must be binary for study {study_id}")
            seen.add(study_id)
            study_ids.append(study_id)
            rows.append(values)
    if not study_ids:
        raise ValueError(f"No annotation rows loaded from {annotation_csv} "
                         f"(required subset is empty?)")
    matrix = torch.tensor(rows, dtype=torch.float32)  # [N, C]

    matrix.share_memory_()
    return label_names, _AnnotationTable(study_ids, matrix)

def get_image_transform(image_size, encoder_type="vit2d"):
    if encoder_type != "vit2d":
        raise ValueError(
            f"LaUER supports only the vit2d image encoder; received encoder_type={encoder_type!r}"
        )
    from torchvision import transforms as T
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class MEETIDataset(Dataset):
    """Load paired ECG views, clinical text, and optional multi-hot concept targets."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_transform=None,
        tokenizer=None,
        max_text_length: int = 128,
        text_separator: str = "[SEP]",
        tgl_report_marker: str = "report:",
        tgl_explanation_marker: str = "explanation:",
        max_tgl_length: int = 512,
        image_size: int = 0,
        index_dir: str = None,
        require_image: bool = True,
        encoder_type: str = "vit2d",
        annotation_csv: str = None,
        split_csv: str = None,
        preloaded_annotations: Optional[Dict[str, torch.Tensor]] = None,
        preloaded_label_names: Optional[List[str]] = None,

        require_labels: bool = True,

        aug_signal: bool = False,
        aug_image: bool = False,
        aug_n_leads: int = 1,
        aug_scale_min: float = 0.85,
        aug_scale_max: float = 1.15,
        aug_noise: float = 0.02,
        aug_shift_samples: int = 250,

        include_tgl_ids: bool = True,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.text_separator = text_separator
        self.tgl_report_marker = tgl_report_marker
        self.tgl_explanation_marker = tgl_explanation_marker
        self.max_tgl_length = max_tgl_length
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.require_image = require_image
        self.encoder_type = encoder_type
        self.annotation_csv = annotation_csv
        self.split_csv = split_csv

        self.require_labels = require_labels

        self.aug_signal = aug_signal
        self.aug_image = aug_image
        self.aug_n_leads = max(1, aug_n_leads)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aug_noise = aug_noise
        self.aug_shift_samples = aug_shift_samples
        self.include_tgl_ids = bool(include_tgl_ids)
        self._rng = np.random.default_rng(None)

        assert tokenizer is not None, "[MEETIDataset] tokenizer must be provided!"
        self.tokenizer = tokenizer

        self.image_transform = image_transform

        idx_dir = _resolve_index_dir(data_root, index_dir)
        samples = _load_split_from_index(data_root, idx_dir, split)

        if samples is not None:
            print(f"[MEETIDataset] Loaded from index: {len(samples)} samples ({split})")

            if require_image and len(samples) > 0:
                probe = samples[:64]
                missing_png = sum(
                    1 for s in probe
                    if s.get('png_path') is None or not os.path.exists(s['png_path'])
                )
                if missing_png > 0:
                    print(f"[MEETIDataset] WARNING: {missing_png}/{len(probe)} probed "
                          f"samples are missing .png files; they will fall back to "
                          f"zero images. Rebuild file_index/ or fix the data tree.")
        else:
            print(f"[MEETIDataset] WARNING: No index at {idx_dir}, falling back to "
                  f"directory scan (SLOW on network disk). Pre-build file_index/ "
                  f"with the old project's build_file_index.py for faster startup.")
            if self.split_csv:
                samples = _scan_and_csv_split(
                    data_root, split, require_image, self.split_csv)
            else:
                samples = _scan_and_hash_split(data_root, split, require_image)
            print(f"[MEETIDataset] Scanned: {len(samples)} samples ({split})")

        self.label_names: Optional[List[str]] = None
        self.annotations: Optional[Dict[str, torch.Tensor]] = None
        if preloaded_annotations is not None:

            self.label_names = preloaded_label_names
            self.annotations = preloaded_annotations
            labeled = [s for s in samples if s['sample_id'] in self.annotations]
            if self.require_labels:
                n_dropped = len(samples) - len(labeled)
                if n_dropped > 0:
                    print(f"[MEETIDataset] {split}: WARNING {n_dropped}/{len(samples)} "
                          f"samples have no annotation rows and were DROPPED")
                samples = labeled
                print(f"[MEETIDataset] {split}: {len(samples)} samples with labels "
                      f"({len(self.label_names)} classes)")
            else:

                n_missing = len(samples) - len(labeled)
                if n_missing > 0:
                    print(f"[MEETIDataset] {split}: keeping all {len(samples)} samples "
                          f"({n_missing} without annotation -> zero labels)")
        elif annotation_csv:
            self.label_names = read_label_names(annotation_csv)
            self.annotations = load_annotations(
                annotation_csv, [s['sample_id'] for s in samples]
            )

            labeled = [s for s in samples if s['sample_id'] in self.annotations]
            if self.require_labels:
                n_dropped = len(samples) - len(labeled)
                if n_dropped > 0:
                    print(f"[MEETIDataset] {split}: WARNING {n_dropped}/{len(samples)} "
                          f"samples have no annotation rows and were DROPPED")
                samples = labeled
                print(f"[MEETIDataset] {split}: {len(samples)} samples with labels "
                      f"({len(self.label_names)} classes)")
            else:
                n_missing = len(samples) - len(labeled)
                if n_missing > 0:
                    print(f"[MEETIDataset] {split}: keeping all {len(samples)} samples "
                          f"({n_missing} without annotation -> zero labels)")

        self.samples = samples

    def _load_mat(self, mat_path: str):
        """Read and validate the ECG signal array from a MATLAB file."""
        try:

            data = loadmat(
                mat_path,
                variable_names=['id', 'signal', 'report', 'LLM_Interpretation'],
            )
            if 'signal' not in data:
                raise ValueError(
                    f"[MEETIDataset] .mat file missing 'signal' field: {mat_path}"
                )

            signal = data['signal'].astype(np.float32)
            if signal.shape != (12, 5000):
                raise ValueError(
                    f"[MEETIDataset] Unexpected signal shape {tuple(signal.shape)} "
                    f"(expected (12, 5000)) in {mat_path}"
                )
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

            report = str(data['report'][0]).strip() if 'report' in data else ""

            llm_text = str(data['LLM_Interpretation'][0]).strip() if 'LLM_Interpretation' in data else ""

            sample_id = str(data['id'][0]) if 'id' in data else ""

            _evict_file_pages(mat_path)
            return signal, report, llm_text, sample_id
        except Exception as e:

            logger.warning(
                f"[MEETIDataset] Failed to load .mat {mat_path}: {e}; "
                f"falling back to zero signal."
            )
            return np.zeros((12, 5000), dtype=np.float32), "", "", ""

    def _load_image(self, png_path: Optional[str]) -> torch.Tensor:
        if png_path is None or not os.path.exists(png_path):
            return self._zero_image()
        try:
            img = Image.open(png_path).convert('RGB')

            _evict_file_pages(png_path)
            if self.image_transform is not None:
                return self.image_transform(img)
            arr = np.asarray(img, dtype=np.uint8)                        # [H, W, 3]
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()   # [3, H, W] uint8
        except Exception as e:
            logger.warning(f"[MEETIDataset] Failed to load image {png_path}: {e}")
            return self._zero_image()

    def _zero_image(self) -> torch.Tensor:
        if self.image_transform is not None:
            return torch.zeros(3, 720, 1000)
        return torch.zeros(3, 720, 1000, dtype=torch.uint8)

    def _augment_view(self, signal, image):
        T = int(self._rng.integers(1, 3))
        trs = self._rng.choice(4, size=T, replace=False)
        shift = 0
        lead_k = None
        scale = 1.0
        noise = 0.0
        for tr in trs:
            if tr == 0:
                lead_k = int(self._rng.integers(12))
            elif tr == 1:
                shift = int(self._rng.integers(
                    -self.aug_shift_samples, self.aug_shift_samples + 1))
            elif tr == 2:
                scale = float(self._rng.uniform(
                    self.aug_scale_min, self.aug_scale_max))
            else:
                noise = float(self.aug_noise)

        sig = signal
        if lead_k is not None:
            sig = sig.copy()
            sig[lead_k, :] = 0.0
        if shift:
            sig = np.roll(sig, shift, axis=1)
        if scale != 1.0:
            sig = sig * scale
        if noise > 0:
            sig = sig + self._rng.normal(0.0, noise, sig.shape)

        img = image
        if shift:
            px = int(round(shift / 5.0))
            img = torch.roll(img, px, dims=2)
        if lead_k is not None:
            img = img.clone()
            img[:, lead_k * 60:(lead_k + 1) * 60, :] = 0.0

        sig = np.asarray(sig, dtype=np.float32)
        return sig, img

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt',
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),          # [L_real]
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }

    def _build_tgl_ids(self, report: str, explanation: str) -> torch.Tensor:
        tok = self.tokenizer
        rep_marker_ids = tok.encode(self.tgl_report_marker, add_special_tokens=False)
        expl_marker_ids = tok.encode(self.tgl_explanation_marker, add_special_tokens=False)
        rep_content = tok.encode(report, add_special_tokens=False)
        expl_content = tok.encode(explanation, add_special_tokens=False)

        base = ([tok.cls_token_id] + rep_marker_ids + rep_content
                + [tok.sep_token_id] + expl_marker_ids)

        budget = self.max_tgl_length - len(base) - 1
        if len(expl_content) > budget:
            expl_content = expl_content[:max(budget, 0)]
        ids = base + expl_content + [tok.sep_token_id]

        while len(ids) > self.max_tgl_length:
            if expl_content:
                expl_content = expl_content[:-1]
                ids = base + expl_content + [tok.sep_token_id]
            elif rep_content:
                rep_content = rep_content[:-1]
                base = ([tok.cls_token_id] + rep_marker_ids + rep_content
                        + [tok.sep_token_id] + expl_marker_ids)
                ids = base + expl_content + [tok.sep_token_id]
            else:
                break
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        info = self.samples[idx]

        signal, report, llm_text, mat_id = self._load_mat(info['mat_path'])

        combined = _build_text(report, llm_text, self.text_separator)

        ecg_signal = torch.from_numpy(signal)              # [12, 5000]
        ecg_image = self._load_image(info['png_path'])     # [3, 720, 1000]
        tokens = self._tokenize_text(combined)

        tgl_ids = (self._build_tgl_ids(report, llm_text)
                       if self.include_tgl_ids else None)

        sample_id = mat_id or info['sample_id']
        result = {
            'ecg_signal': ecg_signal,
            'ecg_image': ecg_image,
            'text_input_ids': tokens['input_ids'],
            'text_attention_mask': tokens['attention_mask'],
            'sample_id': sample_id,
        }
        if tgl_ids is not None:
            result['tgl_text_ids'] = tgl_ids

        if (self.aug_signal or self.aug_image) and self.split == 'train':
            sig_aug, img_aug = self._augment_view(signal, ecg_image)
            result['ecg_signal_aug'] = torch.from_numpy(sig_aug)
            result['ecg_image_aug'] = img_aug

        if self.annotations is not None:
            labels = self.annotations.get(info['sample_id'])
            if labels is None:
                labels = torch.zeros(len(self.label_names), dtype=torch.float32)
            result['labels'] = labels
        return result

# Collate Function & DataLoader Factory

def meeti_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    result = {
        'ecg_signal': torch.stack([b['ecg_signal'] for b in batch]),
        'ecg_image': torch.stack([b['ecg_image'] for b in batch]),
        'text_input_ids': pad_sequence(
            [b['text_input_ids'] for b in batch],
            batch_first=True, padding_value=0,   # 0 = PAD token id
        ),
        'text_attention_mask': pad_sequence(
            [b['text_attention_mask'] for b in batch],
            batch_first=True, padding_value=0,
        ),
        'sample_id': [b['sample_id'] for b in batch],
    }

    if 'tgl_text_ids' in batch[0]:
        result['tgl_text_ids'] = pad_sequence(
            [b['tgl_text_ids'] for b in batch],
            batch_first=True, padding_value=0,
        )
    if 'labels' in batch[0]:
        result['labels'] = torch.stack([b['labels'] for b in batch])
    if 'ecg_signal_aug' in batch[0]:
        result['ecg_signal_aug'] = torch.stack([b['ecg_signal_aug'] for b in batch])
        result['ecg_image_aug'] = torch.stack([b['ecg_image_aug'] for b in batch])
    return result

def create_meeti_dataloader(
    dataset_cfg, train_cfg, tokenizer,
    max_text_length: int, image_size: int,
    split: str = "train",
    image_transform=None,
    index_dir: str = None,
    encoder_type: str = "vit2d",
    annotation_csv: str = None,
    split_csv: str = None,
    preloaded_annotations: Optional[Dict[str, torch.Tensor]] = None,
    preloaded_label_names: Optional[List[str]] = None,
    require_labels: bool = True,
    include_tgl_ids: bool = True,
) -> DataLoader:
    is_train = (split == 'train')

    if is_train:
        nw, pm, pf, persistent, shuffle, dl = dataset_cfg.get_train_loader_cfg()
        bs = train_cfg.batch_size
    else:
        bs, nw, pm, persistent, shuffle, dl = dataset_cfg.get_val_loader_cfg()
        pf = None

    dataset = MEETIDataset(
        data_root=dataset_cfg.data_root,
        split=split,
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_text_length=max_text_length,
        text_separator=dataset_cfg.text_separator,
        tgl_report_marker=getattr(dataset_cfg, 'tgl_report_marker', 'report:'),
        tgl_explanation_marker=getattr(dataset_cfg, 'tgl_explanation_marker', 'explanation:'),
        max_tgl_length=getattr(dataset_cfg, 'max_tgl_length', 512),
        image_size=image_size,
        index_dir=index_dir,
        require_image=True,
        encoder_type=encoder_type,
        annotation_csv=annotation_csv,
        split_csv=split_csv,
        preloaded_annotations=preloaded_annotations,
        preloaded_label_names=preloaded_label_names,
        require_labels=require_labels,
        include_tgl_ids=include_tgl_ids,

        aug_signal=getattr(dataset_cfg, 'aug_signal', False),
        aug_image=getattr(dataset_cfg, 'aug_image', False),
        aug_n_leads=getattr(dataset_cfg, 'aug_n_leads', 1),
        aug_scale_min=getattr(dataset_cfg, 'aug_scale_min', 0.85),
        aug_scale_max=getattr(dataset_cfg, 'aug_scale_max', 1.15),
        aug_noise=getattr(dataset_cfg, 'aug_noise', 0.02),
        aug_shift_samples=getattr(dataset_cfg, 'aug_shift_samples', 250),
    )

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=pm,
        collate_fn=meeti_collate_fn,
        drop_last=dl,
        worker_init_fn=seed_worker,
        generator=make_data_generator(train_cfg.seed, split=split),

        persistent_workers=persistent,
    )
    if is_train and nw > 0:

        loader_kwargs['prefetch_factor'] = pf
    loader = DataLoader(**loader_kwargs)
    return loader

def _resolve_split_csv(dataset_cfg, split: str) -> str:
    ann = getattr(dataset_cfg, 'annotation_csv', '') or ''
    if split == 'train':
        return ann
    key = 'val_annotation_csv' if split == 'val' else 'test_annotation_csv'
    return (getattr(dataset_cfg, key, '') or '') or ann

def create_meeti_dataloaders(dataset_cfg, train_cfg, tokenizer,
                             max_text_length: int = 512, image_size: int = 0,
                             image_transform=None, index_dir=None,
                             encoder_type: str = "vit2d",
                             load_labels: bool = False,
                             fail_on_error: bool = True,
                             include_tgl_ids: Optional[bool] = None
                             ) -> Dict[str, DataLoader]:
    """Build split-specific loaders; fail immediately when a required split cannot be loaded."""

    idx_dir = _resolve_index_dir(dataset_cfg.data_root, index_dir)
    split_csv_all = {s: _resolve_split_csv(dataset_cfg, s) for s in ['train', 'val', 'test']}
    per_split_ann = {}
    for split in ['train', 'val', 'test']:
        split_csv = split_csv_all[split]
        ann_csv = split_csv if load_labels else ''
        if not ann_csv:
            per_split_ann[split] = (None, None)
            continue
        split_samples = _load_split_from_index(dataset_cfg.data_root, idx_dir, split)
        if split_samples is None:
            split_samples = _scan_and_csv_split(
                dataset_cfg.data_root, split, require_image=True, split_csv=split_csv)
        required_ids = [s['sample_id'] for s in split_samples]
        ann_train = split_csv_all['train']
        print(f"[DataLoader] Loading annotations for {split} from {ann_csv} (single pass)...")
        try:
            names, table = _load_annotations_all(ann_csv, required_ids)
        except ValueError as e:

            if ann_csv != ann_train:
                print(f"[DataLoader] WARNING: {split} annotation CSV has no matching rows "
                      f"({str(e)[:80]}); falling back to {ann_train}")
                names, table = _load_annotations_all(ann_train, required_ids)
            else:
                raise
        print(f"[DataLoader] {split}: {len(table)} annotation rows / "
              f"{len(required_ids)} required ids, {len(names)} classes")
        per_split_ann[split] = (names, table)

    _ref_order = None
    for split in ['train', 'val', 'test']:
        names, _ = per_split_ann[split]
        if names is None:
            continue
        order = tuple(names)
        if _ref_order is None:
            _ref_order = order
        elif order != _ref_order:
            raise ValueError(
                f"[DataLoader] Label column ORDER mismatch: {split} CSV differs "
                f"from the reference (train) order. All split CSVs must share the "
                f"same version-matched canonical label column order."
            )

    loaders = {}
    for split in ['train', 'val', 'test']:
        try:
            preloaded_label_names, preloaded_annotations = per_split_ann[split]
            split_csv = split_csv_all[split]
            loaders[split] = create_meeti_dataloader(
                dataset_cfg, train_cfg, tokenizer, max_text_length, image_size,
                split, image_transform, index_dir, encoder_type=encoder_type,
                split_csv=split_csv,
                annotation_csv=(None if preloaded_annotations is not None
                                else (split_csv if load_labels else None)),
                preloaded_annotations=preloaded_annotations,
                preloaded_label_names=preloaded_label_names,

                require_labels=(split != 'test'),

                include_tgl_ids=(True if include_tgl_ids is None
                                     else include_tgl_ids),
            )
        except Exception as e:
            if fail_on_error:

                raise RuntimeError(
                    f"[DataLoader] Failed to create {split} loader: {e}. "
                    f"Check annotation CSV paths, file indexes, "
                    f"and data-root permissions."
                ) from e
            print(f"[DataLoader] Failed to create {split} loader: {e}")
            loaders[split] = None
    return loaders
