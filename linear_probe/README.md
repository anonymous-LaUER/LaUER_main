# Linear probing with LaUER

Train a linear multilabel classifier on frozen LaUER features using the supplied MELP split manifests. Raw signal/image CLS features have width 192 for the tiny encoders; the fused CLS feature has width 768. The default `--single-modal-feature cls` matches this protocol. Fusion requires CGL or TGL training.

## Run one task

From the repository root:

```bash
python -m linear_probe.run --task ptbxl_super_class --mode fusion --ratio 1.0 --budget melp --encoder-checkpoint output/stage/lauer/sgl_cgl_tgl/best_weights.pt --config configs/pretrain_SGL+CGL+TGL.yaml
```

| Dataset | Task keys |
| --- | --- |
| PTB-XL | `ptbxl_super_class`, `ptbxl_sub_class`, `ptbxl_form`, `ptbxl_rhythm` |
| CPSC2018 | `icbeb` |
| CSN | `chapman` |

Supported modes are `signal_only`, `image_only`, and `fusion`; training-label fractions are `0.01`, `0.1`, and `1.0`. Use the YAML that matches the checkpoint architecture.

## Run a task matrix

```bash
python -m linear_probe.run --matrix --tasks ptbxl_super_class icbeb chapman --modes signal_only image_only fusion --ratios 0.01 0.1 1.0 --budgets melp --seeds 42 --encoder-checkpoint output/stage/lauer/sgl_cgl_tgl/best_weights.pt --config configs/pretrain_SGL+CGL+TGL.yaml
```

Use `--dry-run` to inspect a run setup. Feature extraction requires the underlying datasets and external MedCPT assets. Generated feature caches, probe heads, and metrics are local outputs and are not distributed.

## Paths and preprocessing

Default dataset roots are `datasets/PTB-XL`, `datasets/CPSC2018`, and `datasets/CSN`. Set `LAUER_PTBXL_ROOT`, `LAUER_ICBEB_ROOT`, or `LAUER_CHAPMAN_ROOT` for external locations. `LAUER_LP_CACHE` and `LAUER_LP_OUTPUT` override cache and run directories.

`data_split/` contains split identities and labels, not raw recordings. Its active dataset folders are `CSN`, `CPSC`, and `ptbxl`. CSV labels were rebuilt from source annotations on 2026-10-28; CSN uses exact original header codes, CPSC uses REFERENCE.csv, and ptbxl uses SCP code presence. CSN VB has no positives in validation/test: report its AUROC as unavailable. Legacy CLI task keys remain compatible. The loaders preserve the project's signal preprocessing and lead-order contract; do not normalize or reorder the same signal again outside that pipeline. Feature caches include checkpoint/configuration identity to prevent accidental reuse across models.

## Checks

```bash
python -m pytest linear_probe/tests -q -p no:cacheprovider
python -m linear_probe.run --help
```

Tests that require real datasets skip when those assets are missing. See the [main README](../README.md) for training, weight preparation, and checkpoint compatibility.
