"""Typed configuration for LaUER encoders, SGL/CGL/TGL objectives, data, and training."""
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class ViT1DConfig:
    model_variant: str = "tiny"               # "tiny"|"small"|"middle"|"base" → width=192/384/512/768
    seq_length: int = 5000
    lead_num: int = 12
    patch_size: int = 125
    checkpoint_path: str = ""

    pos_embedding_type: str = "additive"

    dropout: float = 0.1
    drop_path_rate: float = 0.1

    @property
    def width(self) -> int:
        return {"tiny": 192, "small": 384, "middle": 512, "base": 768}[self.model_variant]

    @property
    def depth(self) -> int:
        return 12

    @property
    def heads(self) -> int:
        return {"tiny": 3, "small": 6, "middle": 8, "base": 12}[self.model_variant]

    @property
    def mlp_dim(self) -> int:
        return self.width * 4

    @property
    def num_patches(self) -> int:
        return self.seq_length // self.patch_size

@dataclass
class ECGEncoderConfig:
    encoder_type: str = "vit1d"
    freeze: bool = True
    vit1d: ViT1DConfig = field(default_factory=ViT1DConfig)

    @property
    def effective_hidden_size(self) -> int:
        return self.vit1d.width

@dataclass
class ViT2DConfig:
    model_variant: str = "tiny"               # "tiny"|"small"|"middle"|"base" → width=192/384/512/768
    image_height: int = 720
    image_width: int = 1000
    lead_num: int = 12
    patch_height: int = 60
    patch_width: int = 25
    checkpoint_path: str = ""

    pos_embedding_type: str = "additive"

    dropout: float = 0.1
    drop_path_rate: float = 0.1

    @property
    def width(self) -> int:
        return {"tiny": 192, "small": 384, "middle": 512, "base": 768}[self.model_variant]

    @property
    def depth(self) -> int:
        return 12

    @property
    def heads(self) -> int:
        return {"tiny": 3, "small": 6, "middle": 8, "base": 12}[self.model_variant]

    @property
    def mlp_dim(self) -> int:
        return self.width * 4

    @property
    def num_patches_per_lead(self) -> int:
        return self.image_width // self.patch_width

    @property
    def num_patches_total(self) -> int:
        return self.lead_num * self.num_patches_per_lead

@dataclass
class ImageEncoderConfig:
    encoder_type: str = "vit2d"
    freeze: bool = True
    vit2d: ViT2DConfig = field(default_factory=ViT2DConfig)

    @property
    def hidden_size(self) -> int:
        return self.vit2d.width

    @property
    def image_size(self) -> int:
        return 0

@dataclass
class TextEncoderConfig:
    model_name_or_path: str = ""
    query_model_name_or_path: str = ""
    tokenizer_name_or_path: str = ""
    hidden_size: int = 768
    max_length: int = 512

    freeze: bool = True

@dataclass
class ModalityMLPConfig:
    input_dim: int = 768
    output_dim: int = 768
    num_layers: int = 1
    hidden_dim: int = 0
    activation: str = "none"       # "gelu" | "none"
    layernorm: float = 0.0
    layernorm_position: str = "hidden_pre"  # hidden_pre|hidden_post|output|both_pre|both_post
    dropout: float = 0.0

@dataclass
class SGLAdapterConfig:
    proj1: ModalityMLPConfig = field(default_factory=ModalityMLPConfig)  # tokens→D_mid
    proj2: ModalityMLPConfig = field(default_factory=ModalityMLPConfig)  # pool→out
    pooling: str = "cls"              # "cls" | "mean" | "attn" | "cls+attn" | "lead_attn" 

    lead_num: int = 12

    lead_pooling_dim: int = 48

@dataclass
class CGLAdapterConfig:
    proj1: ModalityMLPConfig = field(default_factory=ModalityMLPConfig)  # tokens→D_mid

@dataclass
class ModalityAdapterConfig:
    sgl: SGLAdapterConfig = field(default_factory=SGLAdapterConfig)
    cgl: CGLAdapterConfig = field(default_factory=CGLAdapterConfig)

@dataclass
class WeightDecaySwitches:
    switches: Tuple[bool, ...] = (True, True, True, True, True, False, False, True)

    @property
    def proj1_sgl(self) -> bool: return self.switches[0]
    @property
    def proj1_cgl(self) -> bool: return self.switches[1]
    @property
    def proj2(self) -> bool: return self.switches[2]
    @property
    def fusion_encoder(self) -> bool: return self.switches[3]
    @property
    def cgl_head(self) -> bool: return self.switches[4]
    @property
    def image_encoder(self) -> bool: return self.switches[5]
    @property
    def signal_encoder(self) -> bool: return self.switches[6]
    @property
    def tgl(self) -> bool: return self.switches[7]

@dataclass
class ProjectionConfig:
    ecg: ModalityAdapterConfig = field(default_factory=ModalityAdapterConfig)
    image: ModalityAdapterConfig = field(default_factory=ModalityAdapterConfig)
    text: ModalityAdapterConfig = field(default_factory=ModalityAdapterConfig)
    tgl_proj: 'TGLAdapterConfig' = field(default_factory=lambda: TGLAdapterConfig())
    use_shared_proj1: bool = False

@dataclass
class SGLConfig:
    """Semantic alignment temperature, pair weights, and global objective weight."""
    temperature: float = 0.07
    learnable_temp: bool = True
    temp_max: float = 100.0

    loss_weights: Tuple[float, float, float] = (1.0, 1.0, 0.5)

    sgl_in_global: float = 1.0

@dataclass
class FusionConfig:
    """Alternating signal-image fusion layout within CGL."""
    num_layers: int = 0
    simple_num_layers: int = 1
    layer1_primary: str = "ecg"                   # Layer 1 primary: "ecg" or "image"
    cross_kv_sequence: Tuple[str, ...] = ("image",)
    detach_cross_kv: Tuple[bool, ...] = (False,)

@dataclass
class TracingHeadConfig:
    """Query-guided tracing depth, attention, and multilabel loss settings."""
    num_layers: int = 2
    num_heads: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.1
    query_proj_hidden: int = 768
    query_self_attn: bool = True
    query_proj: bool = True

    loss_type: str = "bce"
    asl_gamma_pos: float = 0.0
    asl_gamma_neg: float = 2.0
    asl_clip: float = 0.05
    asl_eps: float = 1e-8

    def __post_init__(self):
        if self.loss_type not in ("bce", "asl"):
            raise ValueError(
                f"cgl.multilabel.loss_type must be 'bce' or 'asl': got {self.loss_type!r}"
            )

@dataclass
class CGLConfig:
    """Fusion, matching, and disease-query supervision settings."""

    hidden_size: int = 768
    num_heads: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.1

    cgl_head_dropout: float = 0.1

    cgl_in_global: float = 1.0

    fusion: FusionConfig = field(default_factory=FusionConfig)

    multilabel: TracingHeadConfig = field(default_factory=TracingHeadConfig)

    loss_weights: Tuple[float, float] = (1.0, 1.0)

@dataclass
class TGLAdapterConfig:
    """Projection from fused ECG memory to text-decoder width."""
    enable: bool = False
    input_dim: int = 768
    hidden_dim: int = 768
    output_dim: int = 768
    num_layers: int = 2
    activation: str = "none"     # "gelu" | "none"
    layernorm: float = 0.0
    layernorm_position: str = "hidden_pre"  # hidden_pre|hidden_post|output|both_pre|both_post
    dropout: float = 0.0
    allow_tgl_grad_to_encoder: bool = False

@dataclass
class TGLConfig:
    """Token-granularity text-decoder architecture and generation settings."""

    hidden_size: int = 768
    vocab_size: int = 30522
    num_layers: int = 4
    num_heads: int = 12
    ffn_size: int = 3072
    dropout: float = 0.1

    report_mask_rate: float = 0.95
    mlm_loss_weight: float = 1.0
    ar_loss_weight: float = 0.5
    full_mask_prob: float = 0.2
    full_mask_rampup: bool = False
    lm_weight: float = 1.0

    mode: str = ""
    loss_mode: str = "hybrid"
    region_weights: Tuple[float, float] = (0.5, 0.5)

    @property
    def effective_loss_mode(self) -> str:
        m = self.mode if self.mode else self.loss_mode
        return "ar" if m == "ar" else "hybrid"

    max_generation_len: int = 256
    temperature: float = 1.0
    keyword_threshold: float = 0.9

    medcpt_path: str = "weights/MedCPT-Article-Encoder"

@dataclass
class TrainLoaderConfig:
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None
    prefetch_factor: Optional[int] = None
    persistent_workers: Optional[bool] = None
    shuffle: Optional[bool] = None
    drop_last: Optional[bool] = None

    release_workers_before_val: Optional[bool] = None

@dataclass
class EvalLoaderConfig:
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None
    persistent_workers: Optional[bool] = None
    shuffle: Optional[bool] = None
    drop_last: Optional[bool] = None

@dataclass
class DataConfig:
    data_root: str = ""
    annotation_csv: str = ""
    val_annotation_csv: str = ""
    test_annotation_csv: str = ""
    eval_batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True

    train_prefetch_factor: int = 1
    drop_last: bool = True

    train_loader: TrainLoaderConfig = field(default_factory=TrainLoaderConfig)
    val_loader: EvalLoaderConfig = field(default_factory=EvalLoaderConfig)

    def get_train_loader_cfg(self):
        tl = self.train_loader
        nw = tl.num_workers if tl.num_workers is not None else self.num_workers
        pm = tl.pin_memory if tl.pin_memory is not None else self.pin_memory
        pf = tl.prefetch_factor if tl.prefetch_factor is not None else self.train_prefetch_factor
        persistent = tl.persistent_workers if tl.persistent_workers is not None else (nw > 0)
        shuffle = tl.shuffle if tl.shuffle is not None else True
        dl = tl.drop_last if tl.drop_last is not None else self.drop_last
        return nw, pm, pf, persistent, shuffle, dl

    def get_val_loader_cfg(self):
        vl = self.val_loader
        bs = vl.batch_size if vl.batch_size is not None else self.eval_batch_size
        nw = vl.num_workers if vl.num_workers is not None else 0
        pm = vl.pin_memory if vl.pin_memory is not None else False
        persistent = vl.persistent_workers if vl.persistent_workers is not None else (nw > 0)
        shuffle = vl.shuffle if vl.shuffle is not None else False
        dl = vl.drop_last if vl.drop_last is not None else False
        return bs, nw, pm, persistent, shuffle, dl

    text_separator: str = " [SEP] "

    tgl_report_marker: str = "report:"
    tgl_explanation_marker: str = "explanation:"
    max_tgl_length: int = 512

    aug_signal: bool = False
    aug_image: bool = False
    aug_n_leads: int = 1
    aug_scale_min: float = 0.85
    aug_scale_max: float = 1.15
    aug_noise: float = 0.02
    aug_shift_samples: int = 250

@dataclass
class TrainConfig:

    expected_mode: str = ""
    epochs: int = 30
    batch_size: int = 32
    accumulation_steps: int = 1
    lr: float = 1e-4
    min_lr: float = 0.0
    weight_decay: float = 0.05
    norm_weight_decay: Optional[float] = None
    wd_switches: WeightDecaySwitches = field(default_factory=WeightDecaySwitches)
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8

    scheduler: str = "cosine"                  # "cosine" / "linear" / "constant" / "step"
    warmup_steps: int = 1000
    warmup_ratio: float = 0.0

    grad_clip_norm: float = 1.0
    fp16: bool = True
    bf16: bool = False
    amp_init_scale: float = 4096.0
    amp_growth_interval: int = 8000
    amp_backoff_factor: float = 0.5
    nan_log_interval: int = 100
    nan_debug: bool = False

    seed: int = 42
    deterministic: bool = False
    benchmark: bool = True

    val_every_n_epochs: int = 1
    early_stop_patience: int = 10
    composite_zeroshot_weight: float = 0.5
    composite_cgl_weight: float = 0.5
    composite_mlm_weight: float = 0.0
    composite_ar_weight: float = 0.0
    composite_mlc_weight: float = 0.0
    tgl_ablation_every_n: int = 0

    lr_scale_sgl: float = 0.01
    lr_scale_cgl: float = 1.0
    lr_scale_tgl: float = 1.0
    lr_scale_encoder: float = 0.001

    output_dir: str = "./output"
    run_name: str = "checkpoint"

@dataclass
class LaUERConfig:
    """Configuration shared by LaUER training, evaluation, and downstream inference."""
    ecg_encoder: ECGEncoderConfig = field(default_factory=ECGEncoderConfig)
    image_encoder: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    sgl: SGLConfig = field(default_factory=SGLConfig)
    cgl: CGLConfig = field(default_factory=CGLConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tgl: TGLConfig = field(default_factory=TGLConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):

        #              tokens → proj1_cgl → fusion tokens    (CGL)

        def _sgl_proj1_input(enc_dim, pooling, lead_num, lead_k):
            if pooling == "lead_attn":

                return enc_dim + lead_num * lead_k
            if pooling == "cls+attn":
                return 2 * enc_dim
            return enc_dim

        ecg_hidden = self.ecg_encoder.vit1d.width
        self.projection.ecg.sgl.proj1.input_dim = _sgl_proj1_input(
            ecg_hidden, self.projection.ecg.sgl.pooling,
            self.projection.ecg.sgl.lead_num,
            self.projection.ecg.sgl.lead_pooling_dim,
        )
        self.projection.ecg.sgl.proj2.input_dim = (
            self.projection.ecg.sgl.proj1.output_dim
        )
        self.projection.ecg.sgl.lead_num = self.ecg_encoder.vit1d.lead_num
        self.projection.ecg.cgl.proj1.input_dim = ecg_hidden

        img_hidden = self.image_encoder.vit2d.width
        self.projection.image.sgl.proj1.input_dim = _sgl_proj1_input(
            img_hidden, self.projection.image.sgl.pooling,
            self.projection.image.sgl.lead_num,
            self.projection.image.sgl.lead_pooling_dim,
        )
        self.projection.image.sgl.proj2.input_dim = (
            self.projection.image.sgl.proj1.output_dim
        )
        self.projection.image.sgl.lead_num = self.image_encoder.vit2d.lead_num
        self.projection.image.cgl.proj1.input_dim = img_hidden

        if self.projection.text.sgl.pooling == "lead_attn":
            raise ValueError(
                "pooling='lead_attn' requires ECG leads; "
                "use pooling='cls+attn' for text"
            )
        self.projection.text.sgl.proj1.input_dim = _sgl_proj1_input(
            self.text_encoder.hidden_size, self.projection.text.sgl.pooling,
            self.projection.text.sgl.lead_num,
            self.projection.text.sgl.lead_pooling_dim,
        )
        self.projection.text.sgl.proj2.input_dim = (
            self.projection.text.sgl.proj1.output_dim
        )
        self.projection.text.cgl.proj1.input_dim = self.text_encoder.hidden_size

        if not self.text_encoder.tokenizer_name_or_path:
            self.text_encoder.tokenizer_name_or_path = self.text_encoder.model_name_or_path

def load_config_from_yaml(yaml_path: str) -> LaUERConfig:
    """Load nested YAML settings and infer adapter input widths from encoder dimensions."""
    import yaml
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)

    _FIELD_ALIASES = {
        'text_encoder': {'model_name': 'model_name_or_path', 'tokenizer_name': 'tokenizer_name_or_path'},
        'sgl': {'itc_in_global': 'sgl_in_global'},
        'cgl': {'itm_mode': 'cgl_mode', 'itm_stage': 'cgl_stage', 'itm_head_dropout': 'cgl_head_dropout'},
        'train': {'composite_itm_weight': 'composite_cgl_weight'},
    }

    _NESTED_ALIASES = {}
    _LIST_TO_TUPLE_KEYS = {'loss_weights', 'cross_kv_sequence', 'detach_cross_kv', 'region_weights'}

    def _set_nested_dataclass(dc_instance, yaml_dict):
        for k, v in yaml_dict.items():
            if not hasattr(dc_instance, k):

                warnings.warn(
                    f"[Config] Unknown nested config key ignored: {k!r} "
                    f"(section {type(dc_instance).__name__})",
                    UserWarning, stacklevel=3,
                )
                continue
            attr = getattr(dc_instance, k)
            if hasattr(attr, '__dataclass_fields__') and isinstance(v, dict):
                _set_nested_dataclass(attr, v)
            else:
                if k in _LIST_TO_TUPLE_KEYS and isinstance(v, list):
                    v = tuple(v)
                setattr(dc_instance, k, v)

    itm_raw = cfg_dict.get('cgl') or cfg_dict.get('itm')
    if itm_raw is not None:

        old_mode = itm_raw.get('cgl_mode') or itm_raw.get('itm_mode', '')
        if old_mode in ('2stage', '3stage'):
            itm_raw['cgl_stage'] = old_mode  # "2stage" or "3stage"
            itm_raw['cgl_mode'] = 'cascade'

        for old_k in ('itm_mode', 'itm_stage', 'itm_head_dropout'):
            itm_raw.pop(old_k, None)

        _OLD_TO_CASCADE = {
            'fusion_layers_sit': 'fusion_layers_sit',
            'fusion_layers_sit2': 'fusion_layers_sit2',
            'fusion_layers_ts': 'fusion_layers_ts',
        }
        has_old_cascade_keys = any(old_key in itm_raw for old_key in _OLD_TO_CASCADE)
        if has_old_cascade_keys:
            cascade_raw = itm_raw.get('cascade', {})
            for old_key, new_key in _OLD_TO_CASCADE.items():
                if old_key in itm_raw:
                    cascade_raw.setdefault(new_key, itm_raw[old_key])

            if 'loss_weights' in itm_raw and 'loss_weights' not in cascade_raw:
                cascade_raw['loss_weights'] = itm_raw['loss_weights']
            if cascade_raw:
                itm_raw['cascade'] = cascade_raw

        if 'fusion_layers_si' in itm_raw:
            fusion_raw = itm_raw.get('fusion', {})
            fusion_raw.setdefault('simple_num_layers', itm_raw['fusion_layers_si'])
            itm_raw['fusion'] = fusion_raw

    if 'image_encoder' in cfg_dict:
        ie_raw = cfg_dict['image_encoder']
        encoder_type = ie_raw.get('encoder_type', 'clip')
        has_clip_sub = 'clip' in ie_raw and isinstance(ie_raw.get('clip'), dict)
        has_mae_sub = 'mae' in ie_raw and isinstance(ie_raw.get('mae'), dict)

        if 'model_name' in ie_raw and 'model_name_or_path' not in ie_raw:
            ie_raw['model_name_or_path'] = ie_raw.pop('model_name')

        _OLD_MAE_MAP = {
            'mae_checkpoint': 'checkpoint', 'mae_embed_dim': 'embed_dim',
            'mae_patch_size': 'patch_size', 'mae_depth': 'depth',
            'mae_num_heads': 'num_heads', 'mae_has_cls_token': 'has_cls_token',
            'mae_dropout': 'dropout',
        }

        if not has_clip_sub:
            clip_raw = {}
            for old_key in ('model_name_or_path', 'select_layer'):
                if old_key in ie_raw:
                    clip_raw[old_key] = ie_raw.pop(old_key)

            if encoder_type != 'mae':
                for key in ('hidden_size', 'image_size'):
                    if key in ie_raw:
                        clip_raw[key] = ie_raw.pop(key)
            if clip_raw:
                ie_raw['clip'] = clip_raw

        if not has_mae_sub:
            mae_raw = {}
            for old_key, new_key in _OLD_MAE_MAP.items():
                if old_key in ie_raw:
                    mae_raw[new_key] = ie_raw.pop(old_key)
            if encoder_type == 'mae' and 'image_size' in ie_raw:
                mae_raw['image_size'] = ie_raw.pop('image_size')
            if mae_raw:
                ie_raw['mae'] = mae_raw

        ie_raw.pop('hidden_size', None)
        ie_raw.pop('image_size', None)

        for old_key in _OLD_MAE_MAP:
            ie_raw.pop(old_key, None)

    _SECTION_ALIASES = {'itc': 'sgl', 'itm': 'cgl'}
    for old_name, new_name in _SECTION_ALIASES.items():
        if old_name in cfg_dict and new_name not in cfg_dict:
            cfg_dict[new_name] = cfg_dict.pop(old_name)

    if 'tgl' in cfg_dict and 'tgl_proj' in cfg_dict['tgl']:
        if 'projection' not in cfg_dict:
            cfg_dict['projection'] = {}
        if 'tgl_proj' not in cfg_dict['projection']:
            cfg_dict['projection']['tgl_proj'] = cfg_dict['tgl'].pop('tgl_proj')
        else:
            cfg_dict['tgl'].pop('tgl_proj', None)

    if 'projection' in cfg_dict:
        proj = cfg_dict['projection']
        for modality in ['ecg', 'image', 'text']:
            if modality not in proj:
                continue
            m = proj[modality]

            if 'proj1' in m and 'sgl' not in m and 'cgl' not in m:
                old_proj1 = m.pop('proj1', {})
                old_proj2 = m.pop('proj2', {})
                old_pooling = m.pop('pooling', 'cls')
                m.pop('ecg_sgl_path', None)

                m['sgl'] = {
                    'proj1': old_proj1,
                    'proj2': old_proj2,
                    'pooling': old_pooling if old_pooling else 'cls',
                }
                m['cgl'] = {
                    'proj1': old_proj1,
                }

        def _patch_old_mlp(mlp_dict):
            if not isinstance(mlp_dict, dict):
                return
            if 'activation' not in mlp_dict and 'layernorm' not in mlp_dict:
                if mlp_dict.get('num_layers', 1) >= 2 and mlp_dict.get('dropout', 0) > 0:
                    mlp_dict.setdefault('activation', 'gelu')

            for sub_key in ('proj1', 'proj2'):
                if sub_key in mlp_dict and isinstance(mlp_dict[sub_key], dict):
                    sub = mlp_dict[sub_key]
                    if 'activation' not in sub and 'layernorm' not in sub:
                        if sub.get('num_layers', 1) >= 2 and sub.get('dropout', 0) > 0:
                            sub.setdefault('activation', 'gelu')

        _patch_old_mlp(proj.get('tgl_proj', {}))
        for modality in ['ecg', 'image', 'text']:
            m = proj.get(modality, {})
            for path in ['sgl', 'cgl']:
                _patch_old_mlp(m.get(path, {}))

    config = LaUERConfig()
    for section_name in ['ecg_encoder', 'image_encoder', 'text_encoder', 'projection',
                          'sgl', 'cgl', 'tgl', 'data', 'train']:
        if section_name not in cfg_dict:
            continue
        section = getattr(config, section_name)
        aliases = _FIELD_ALIASES.get(section_name, {})
        for k, v in cfg_dict[section_name].items():
            actual_k = aliases.get(k, k)
            if not hasattr(section, actual_k):

                warnings.warn(
                    f"[Config] Unknown config key ignored: {section_name}.{k!r}",
                    UserWarning, stacklevel=3,
                )
                continue
            attr = getattr(section, actual_k)
            # wd_switches: YAML list → tuple → WeightDecaySwitches.switches
            if actual_k == 'wd_switches' and isinstance(v, list):
                if len(v) != 8:
                    raise ValueError(
                        f"train.wd_switches must have exactly 8 entries "
                        f"(v3: proj1_sgl/proj1_cgl/proj2/fusion_encoder/"
                        f"cgl_head/image_encoder/signal_encoder/tgl), "
                        f"got {len(v)}: {v}. Old 9-entry configs are not supported."
                    )
                attr.switches = tuple(v)
            elif hasattr(attr, '__dataclass_fields__') and isinstance(v, dict):
                _set_nested_dataclass(attr, v)
            else:
                if k in _LIST_TO_TUPLE_KEYS and isinstance(v, list):
                    v = tuple(v)
                setattr(section, actual_k, v)

    config.__post_init__()
    return config
