"""Command-line LaUER report and explanation generation from paired ECG signal and image."""
import os, sys, argparse
from dataclasses import asdict, is_dataclass
from collections.abc import Mapping
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from configs.config import load_config_from_yaml
from models.lauer import LaUER

def _config_mapping(config):
    if is_dataclass(config):
        return asdict(config)
    return config if isinstance(config, Mapping) else {}

def _projection_architecture(config, source_mode="sgl+cgl+tgl"):
    cfg = _config_mapping(config)
    projection = cfg.get('projection', {})
    if not isinstance(projection, Mapping):
        return {}

    rank = {
        'sgl': 1,
        'sgl+cgl': 2,
        'sgl+cgl+tgl': 3,
    }.get(source_mode, 3)
    result = {}

    def add_mlp(prefix, mlp):
        if not isinstance(mlp, Mapping):
            return
        try:
            num_layers = int(mlp.get('num_layers', 1))
            input_dim = int(mlp.get('input_dim', 768))
            output_dim = int(mlp.get('output_dim', 768))
            hidden_dim = int(mlp.get('hidden_dim', 0))
        except (TypeError, ValueError):
            return

        resolved_hidden = hidden_dim
        if num_layers >= 2 and resolved_hidden <= 0:
            resolved_hidden = (input_dim + output_dim) // 2

        result[f'{prefix}.input_dim'] = input_dim
        result[f'{prefix}.output_dim'] = output_dim
        result[f'{prefix}.num_layers'] = num_layers
        if num_layers >= 2:
            result[f'{prefix}.hidden_dim'] = resolved_hidden
        result[f'{prefix}.activation'] = str(
            mlp.get('activation', 'none')
        ).lower()

        try:
            layernorm_eps = float(mlp.get('layernorm', 0.0))
        except (TypeError, ValueError):
            layernorm_eps = 0.0
        result[f'{prefix}.layernorm_enabled'] = layernorm_eps > 0
        if layernorm_eps > 0:
            result[f'{prefix}.layernorm_position'] = str(
                mlp.get('layernorm_position', 'hidden_pre')
            ).lower()
            result[f'{prefix}.layernorm_eps'] = layernorm_eps

        try:
            dropout = float(mlp.get('dropout', 0.0))
        except (TypeError, ValueError):
            dropout = 0.0
        result[f'{prefix}.dropout_enabled'] = dropout > 0

    for modality in ('ecg', 'image', 'text'):
        modality_cfg = projection.get(modality, {})
        if not isinstance(modality_cfg, Mapping):
            continue
        sgl_cfg = modality_cfg.get('sgl', {})
        if isinstance(sgl_cfg, Mapping):
            add_mlp(
                f'projection.{modality}.sgl.proj1',
                sgl_cfg.get('proj1', {}),
            )
            add_mlp(
                f'projection.{modality}.sgl.proj2',
                sgl_cfg.get('proj2', {}),
            )
            if 'pooling' in sgl_cfg:
                result[f'projection.{modality}.sgl.pooling'] = str(
                    sgl_cfg['pooling']
                ).lower()

        if rank >= 2:
            cgl_cfg = modality_cfg.get('cgl', {})
            if isinstance(cgl_cfg, Mapping):
                add_mlp(
                    f'projection.{modality}.cgl.proj1',
                    cgl_cfg.get('proj1', {}),
                )

    if rank >= 2 and 'use_shared_proj1' in projection:
        result['projection.use_shared_proj1'] = bool(
            projection['use_shared_proj1']
        )

    if rank >= 3:
        tgl_proj = projection.get('tgl_proj', {})
        if isinstance(tgl_proj, Mapping):
            enabled = bool(tgl_proj.get('enable', False))
            try:
                tgl_layers = int(tgl_proj.get('num_layers', 0))
            except (TypeError, ValueError):
                tgl_layers = 0

            if not enabled or tgl_layers == 0:
                result['projection.tgl_proj.kind'] = 'identity'
            else:
                result['projection.tgl_proj.kind'] = 'mlp'
                add_mlp('projection.tgl_proj', tgl_proj)

    return result

def warn_projection_config_mismatch(config, checkpoint):
    if not isinstance(checkpoint, Mapping):
        print('[Projection Config] No checkpoint metadata; skipped config comparison.')
        return
    saved_config = checkpoint.get('config')
    if not isinstance(saved_config, Mapping):
        print('[Projection Config] Legacy checkpoint without saved config; '
              'using the supplied YAML and state_dict checks only.')
        return

    source_mode = (
        checkpoint.get('source_mode')
        or checkpoint.get('mode')
        or 'sgl+cgl+tgl'
    )
    yaml_spec = _projection_architecture(config, source_mode)
    saved_spec = _projection_architecture(saved_config, source_mode)

    common_paths = sorted(set(yaml_spec) & set(saved_spec))
    mismatches = [
        (path, yaml_spec[path], saved_spec[path])
        for path in common_paths
        if yaml_spec[path] != saved_spec[path]
    ]
    if not mismatches:
        print('[Projection Config] YAML matches checkpoint projection metadata.')
        return

    print(f'[Projection Config] WARNING: found {len(mismatches)} projection '
          'configuration mismatch(es):')
    for path, yaml_value, saved_value in mismatches:
        print(f'  - {path}: YAML={yaml_value!r} | checkpoint={saved_value!r}')
    print('[Projection Config] Continuing with the YAML-built model; '
          'state_dict compatibility will be checked next.')

def warn_projection_state_mismatch(model, state_dict):
    if not isinstance(state_dict, Mapping):
        return
    prefixes = ('adapters.', 'tgl_proj.')
    current_state = model.state_dict()
    expected = {
        key for key in current_state if key.startswith(prefixes)
    }
    provided = {
        key for key in state_dict if key.startswith(prefixes)
    }
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    shape_mismatches = sorted(
        (key, tuple(current_state[key].shape), tuple(state_dict[key].shape))
        for key in expected & provided
        if hasattr(state_dict[key], 'shape')
        and tuple(current_state[key].shape) != tuple(state_dict[key].shape)
    )

    if not (missing or unexpected or shape_mismatches):
        print('[Projection Weights] Projection parameter keys/shapes match.')
        return

    print('[Projection Weights] WARNING: projection state_dict mismatch detected:')
    for key in missing:
        print(f'  - missing in checkpoint: {key}')
    for key in unexpected:
        print(f'  - unexpected in checkpoint: {key}')
    for key, yaml_shape, checkpoint_shape in shape_mismatches:
        print(f'  - shape mismatch: {key}: '
              f'YAML model={yaml_shape} | checkpoint={checkpoint_shape}')

def validate_inference_checkpoint(missing, unexpected):

    critical_prefixes = (
        'adapters.', 'sgl_loss.', 'fusion_', 'cgl_loss_module.',
        'query_guided_tracing_head.', 'tgl.', 'tgl_proj.',
    )
    critical_missing = [
        k for k in missing if k.startswith(critical_prefixes)
    ]
    if critical_missing:
        preview = ', '.join(critical_missing[:8])
        raise RuntimeError(
            f"Checkpoint is missing {len(critical_missing)} critical inference "
            f"parameters: {preview}"
        )
    if missing:

        non_frozen_missing = [
            k for k in missing
            if not k.startswith(("query_encoder.", "text_encoder."))
        ]
        if non_frozen_missing:
            print(f"  Warning: {len(non_frozen_missing)} non-critical missing keys: "
                  f"{non_frozen_missing[:5]}...")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys: {unexpected[:5]}...")

def load_and_preprocess_image(image_path: str, encoder_type: str = "vit2d") -> torch.Tensor:
    if encoder_type != "vit2d":
        raise ValueError(
            f"LaUER supports only the vit2d image encoder; received encoder_type={encoder_type!r}"
        )
    img = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return transform(img).unsqueeze(0)  # [1, 3, H, W]

def load_signal(mat_path: str) -> torch.Tensor:
    import scipy.io as sio
    data = sio.loadmat(mat_path)

    for key in ['val', 'ecg', 'signal', 'ecg_signal', 'data']:
        if key in data:
            signal = data[key]
            break
    else:

        signal = None
        for v in data.values():
            if isinstance(v, np.ndarray) and v.size > 100:
                signal = v
                break
        if signal is None:
            raise ValueError(f"Cannot find ECG signal in {mat_path}. Keys: {list(data.keys())}")

    signal = signal.astype(np.float32)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    if signal.shape == (5000, 12):
        signal = signal.T
    elif signal.shape == (1, 12, 5000):
        signal = signal.squeeze(0)
    elif signal.shape == (5000,):
        signal = signal.reshape(1, -1).repeat(12, axis=0)

    assert signal.shape == (12, 5000), f"Expected (12, 5000), got {signal.shape}"
    return torch.from_numpy(signal).unsqueeze(0)  # [1, 12, 5000]

def _matlab_text(value) -> str:
    """Convert a scalar/string MATLAB value loaded by scipy into plain text."""
    while isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.dtype.kind in {"U", "S"} and value.size > 1:
            return "".join(str(item) for item in value.reshape(-1)).strip()
        if value.size != 1:
            return str(value.squeeze()).strip()
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()

def load_mat_text_fields(mat_path: str):
    """Load the optional report and long interpretation stored in an ECG MAT file."""
    import scipy.io as sio

    data = sio.loadmat(mat_path)
    report = _matlab_text(data["report"]) if "report" in data else ""
    llm_text = (
        _matlab_text(data["LLM_Interpretation"])
        if "LLM_Interpretation" in data else ""
    )
    return report, llm_text

def main():
    parser = argparse.ArgumentParser(description='lauer End-to-End Inference')
    parser.add_argument('--config', type=str, default='configs/pretrain_SGL+CGL+TGL.yaml',
                        help='Config YAML path')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained stage weights (best_weights.pt)')
    parser.add_argument('--signal', type=str, default='',
                        help='Path to .mat ECG signal file')
    parser.add_argument('--image', type=str, default='',
                        help='Path to .png ECG image file')
    parser.add_argument('--max_len', type=int, default=512,
                        help='Max generation length')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--max_report_len', type=int, default=128,
                        help='(ignored in autoregressive mode)')
    parser.add_argument('--min_report_len', type=int, default=5,
                        help='(ignored in autoregressive mode)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda / cpu')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    print(f"[Config] Loading: {args.config}")
    config = load_config_from_yaml(args.config)

    encoder_type = getattr(config.image_encoder, 'encoder_type', 'vit2d')
    print(f"[Config] Image encoder: {encoder_type}")

    print(f"[Model] Creating LaUER (sgl+cgl+tgl)...")
    model = LaUER(config, mode='sgl+cgl+tgl')
    model.load_pretrained_encoders(device=str(device))
    model = model.to(device)

    print(f"[Checkpoint] Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt
    warn_projection_config_mismatch(config, ckpt)
    warn_projection_state_mismatch(model, sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    validate_inference_checkpoint(missing, unexpected)
    model.eval()
    print(f"[Model] Loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    if args.signal and args.image:

        run_inference(model, args.signal, args.image,
                      encoder_type, args.max_report_len, args.max_len,
                      args.min_report_len, args.temperature, device)
    else:

        print("\nEnter signal_path image_path (or 'quit'):")
        while True:
            try:
                inp = input("> ").strip()
                if inp.lower() in ('quit', 'exit', 'q'):
                    break
                parts = inp.split()
                if len(parts) >= 2:
                    signal_path, image_path = parts[0], parts[1]
                    run_inference(model, signal_path, image_path,
                                  encoder_type, args.max_report_len, args.max_len,
                                  args.min_report_len, args.temperature, device)
                else:
                    print("Usage: <signal.mat> <image.png>")
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"Error: {e}")

@torch.no_grad()
def run_inference(model, signal_path, image_path, encoder_type,
                  max_report_len, max_llm_len, min_report_len, temperature, device):
    print(f"\n{'='*60}")
    print(f"Signal: {signal_path}")
    print(f"Image:  {image_path}")

    signal = load_signal(signal_path).to(device)                        # [1, 12, 5000]
    image = load_and_preprocess_image(image_path, encoder_type).to(device)  # [1, 3, H, W]

    print(f"Signal shape: {signal.shape}")
    print(f"Image shape:  {image.shape}")

    cross_kv = model.extract_tgl_cross_kv(signal, image)

    mode_name = 'pure AR' if getattr(model, 'loss_mode_ar', False) else 'hybrid MLM+AR'
    print(f"[Strategy] {mode_name} (max_len={max_llm_len}, temperature={temperature})")
    full_ids, report_ids, llm_ids = model.generate_text(
        cross_kv,
        max_len=max_llm_len,
        temperature=temperature,
    )
    print(f"[Strategy] Generated report_len={report_ids.shape[1]}, "
          f"explanation_len={llm_ids.shape[1]}")

    report_text = model.decode_ids(report_ids, skip_special=True) \
        if report_ids.numel() > 0 else "(empty)"
    llm_text = model.decode_ids(llm_ids, skip_special=True) \
        if llm_ids.numel() > 0 else "(empty)"

    print(f"\n{'─'*40}")
    print(f"[REPORT] ({report_ids.shape[1]} tokens)")
    print(f"  {report_text}")
    print(f"{'─'*40}")
    print(f"[LLM]    ({llm_ids.shape[1]} tokens)")
    print(f"  {llm_text}")
    print(f"{'─'*40}")
    print(f"Full: {full_ids.shape[1]} tokens total")

if __name__ == '__main__':
    main()
