# LaUER attention and Grad-CAM inspection

Inspect SGL semantic alignment, CGL disease-query tracing, and TGL text generation using local LaUER checkpoints. Install `requirements-optional.txt` and prepare the external MedCPT directories first.

## Checkpoint layout

The runner reads stage weights from one directory:

```text
weights/stages/
├── sgl.pt
├── sgl_cgl.pt
└── sgl_cgl_tgl.pt
```

These files must contain the matching stage artifact, including configuration metadata. Select only available stages with `--stages`.

```bash
python -m visualization.attention_grad_cam.run --weights-dir weights/stages --stages sgl_cgl_tgl --dataset meeti --signal /path/to/record.mat --image /path/to/record.png --sample-id example --label-root data/annotations --output output/attention
```

MEETI labels come from `train.csv`, `val.csv`, and `test.csv` under `--label-root`. CSN and CPSC2018 use their own benchmark manifests and label spaces. Use `--help` for dataset selection, target classes, attention layers, generation options, and downstream probe-head paths.

## Build a local sample manifest

Prepare the dataset layout and benchmark splits described in the [main README](../../README.md#downstream-tasks). Run the applicable command from the repository root and adjust the dataset and label paths for your installation.

```bash
python -m visualization.attention_grad_cam.make_cpsc_manifest --root datasets/CPSC2018 --out output/attention/manifests/cpsc.csv
python -m visualization.attention_grad_cam.make_csn_manifest --csn-root datasets/CSN --out output/attention/manifests/csn.csv
python -m visualization.attention_grad_cam.make_meeti_manifest --data-root datasets/MEETI --label-root data/annotations --out output/attention/manifests/meeti.csv
```

Pass a generated file to `--manifest`, choose the corresponding `--dataset`, and select the available `--stages`. The generated paths reflect your local data layout. Keep the manifests under the ignored `output/` directory.

By default, the runner exports each requested attention layer. Outputs can include signal/image overlays, attention matrices, raw arrays, metadata, and a local HTML index. These are inspection artifacts generated on demand. Attention or gradient attribution alone does not establish a clinically validated explanation.

The shared loading, dataset, and attribution helpers are in `visualization/grad_cam/`. No historical plots or attribution scores are included.
