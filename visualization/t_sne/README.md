# t-SNE inspection of LaUER representations

`tsne_cluster.py` reads features already cached by `linear_probe`; it does not load a model or extract raw records itself. First run the relevant linear-probe feature extraction with the same checkpoint, YAML, feature kind, and AMP setting.

```bash
python -m visualization.t_sne.tsne_cluster --dataset ptbxl_rhythm --mode fusion --checkpoint output/stage/lauer/sgl_cgl_tgl/best_weights.pt --config configs/pretrain_SGL+CGL+TGL.yaml --split test
```

Use `--list-classes` to inspect available class labels. Class selection, per-class sampling, point limits, perplexity, and random seed are configurable through `--help`. Install `requirements-optional.txt` for plotting tools.

The default feature kind is CLS. Signal and image features describe their respective encoders; fusion features require a checkpoint trained with CGL. Multilabel records need an explicit display-label policy, so the plotted subset does not represent every original label equally.

Treat t-SNE as an exploratory view of a sampled representation space. It is not a substitute for the held-out classification protocol. The optional 3D rendering helper consumes previously exported local visualization files. No example result plots are distributed.
