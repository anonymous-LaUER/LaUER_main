"""t-SNE clustering visualisation for frozen lauer representations.

Runs on CACHED features produced by linear_probe (no checkpoint loading, no GPU).

    python visualization/t_sne/tsne_cluster.py --dataset all
    python visualization/t_sne/tsne_cluster.py --dataset chapman --classes 2,9,20,31,36
    python visualization/t_sne/tsne_cluster.py --dataset ptbxl_rhythm --list-classes

See visualization/t_sne/README.md for the full parameter reference.

Pipeline
--------
    cache(.pt) --ids--> label CSV --align--> single-label filter --> class selection
        --> kNN metric on ALL selected samples (original feature space)
        --> balanced subsample --> L2 normalise --> PCA --> t-SNE --> scatter + legend
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

from linear_probe.config import PathsConfig, PreprocessConfig, ProbeConfig, get_task  # noqa: E402
from linear_probe.config import TASKS, TASK_ORDER  # noqa: E402
from linear_probe.features import encoder_tag, feature_cache_path, load_feature_set  # noqa: E402
from linear_probe.splits import load_public_splits  # noqa: E402

# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

DATASETS = {key: (TASKS[key].family, TASKS[key].display_name) for key in TASK_ORDER}
DATASET_ORDER = list(TASK_ORDER)

MODES = ("signal_only", "image_only", "fusion")
MODE_LABEL = {"signal_only": "ECG signal", "image_only": "ECG image", "fusion": "Fusion"}

DEFAULT_CHECKPOINT = "weights/lauer_sgl_cgl_tgl.pt"
DEFAULT_CONFIG = "configs/pretrain_SGL+CGL+TGL.yaml"
DEFAULT_AMP = "bf16"
DEFAULT_FEATURE_KIND = "cls"
RESULTS_ROOT = "output"
KNN_MAX_SAMPLES = 20000

def log(msg):
    print("[tsne] %s" % msg, flush=True)

def l2(X):
    Xn = np.asarray(X, dtype=np.float64)
    return Xn / np.clip(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12, None)

def parse_class_list(spec, names):
    out, lookup = [], {n.lower(): i for i, n in enumerate(names)}
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if token.lstrip("-").isdigit():
            idx = int(token)
            if not 0 <= idx < len(names):
                raise SystemExit("class index %d out of range 0..%d" % (idx, len(names) - 1))
            out.append(idx)
        else:
            if token.lower() not in lookup:
                raise SystemExit("unknown class %r. Valid names: %s" % (token, ", ".join(names)))
            out.append(lookup[token.lower()])
    if not out:
        raise SystemExit("--classes parsed to an empty list")
    seen, uniq = set(), []
    for i in out:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq

def palette(n):
    if n <= 10:
        return list(plt.get_cmap("tab10").colors)[:n]
    if n <= 20:
        return list(plt.get_cmap("tab20").colors)[:n]
    base = (list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
            + list(plt.get_cmap("tab20c").colors))
    if n <= len(base):
        return base[:n]
    return [plt.get_cmap("hsv")(i / max(n, 1)) for i in range(n)]

def tsne_kwargs(max_iter):
    import inspect
    params = inspect.signature(TSNE.__init__).parameters
    kw = dict(n_components=2, init="pca", learning_rate="auto")
    kw["max_iter" if "max_iter" in params else "n_iter"] = max_iter
    return kw

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def resolve_cache_tag(checkpoint, config_yaml, amp):
    paths = PathsConfig()
    probe = ProbeConfig(encoder_checkpoint=checkpoint, config_yaml=config_yaml, amp=amp)
    return encoder_tag(probe, paths), paths

def load_features(family, mode, tag, feature_kind, cache_dir):
    prep_tag = PreprocessConfig().tag()
    path = feature_cache_path(cache_dir, family, mode, tag, prep_tag, feature_kind)
    if not Path(path).is_file():
        raise SystemExit("feature cache missing:\n  %s\n  run linear_probe with the same "
                         "checkpoint/config first." % path)
    fs = load_feature_set(Path(path))
    if fs is None:
        raise SystemExit("failed to load feature cache %s" % path)
    return fs

def collect(dataset, mode, split_names, multilabel, feature_kind, tag, paths, return_ids=False):
    family = DATASETS[dataset][0]
    cohort = load_public_splits(get_task(dataset))
    fs = load_features(family, mode, tag, feature_kind, paths.cache_dir)
    pos = {sid: i for i, sid in enumerate(fs.ids)}
    if len(pos) != len(fs.ids):
        raise ValueError("duplicate sample IDs in feature cache")

    rows, ys, sids, dropped = [], [], [], 0
    counts_all = cohort.labels["train"].sum(axis=0)
    for split in split_names:
        labels = cohort.labels[split]
        for sid in cohort.ids[split]:
            if sid not in pos:
                raise ValueError("sample missing from %s cache: %s" % (mode, sid))
            r = cohort.row_index[split][sid]
            if r >= labels.shape[0]:
                raise ValueError("label row out of bounds: %s" % sid)
            lab = labels[r]
            n_pos = int(lab.sum())
            if multilabel == "rarest":
                if n_pos == 0:
                    dropped += 1
                    continue
                cand = np.nonzero(lab)[0]
                cls = int(cand[int(np.argmin(counts_all[cand]))])
            else:
                if n_pos != 1:
                    dropped += 1
                    continue
                cls = int(np.argmax(lab))
            rows.append(pos[sid])
            ys.append(cls)
            sids.append(sid)
    if not rows:
        raise SystemExit("no usable samples for dataset=%s mode=%s" % (dataset, mode))

    X = fs.feats[np.asarray(rows)].detach().cpu().numpy().astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    names = list(cohort.label_names)
    counts = np.bincount(y, minlength=len(names))
    result = (X, y, names, counts, dropped, int(fs.feats.shape[1]))
    return result + (np.asarray(sids),) if return_ids else result

def class_table(dataset, multilabel):
    cohort = load_public_splits(get_task(dataset))
    names = list(cohort.label_names)
    stats = {}
    for label, splits in (("test", ("test",)), ("all", ("train", "val", "test"))):
        y = np.concatenate([cohort.labels[s] for s in splits], axis=0)
        single = y[y.sum(axis=1) == 1]
        stats[label] = single.sum(axis=0).astype(int) if single.size else np.zeros(len(names), int)
    print("dataset = %s   (%d classes)" % (dataset, len(names)))
    print("  %-4s %-22s %10s %10s" % ("idx", "class", "test", "all-splits"))
    for i, n in enumerate(names):
        print("  %-4d %-22s %10d %10d" % (i, n, stats["test"][i], stats["all"][i]))
    print("  counts = samples with exactly one positive label (multilabel mode = %s)" % multilabel)

# ---------------------------------------------------------------------------
# class selection (the only two knobs the user manages)
# ---------------------------------------------------------------------------

def select_classes(args, names, counts):
    notes, C = [], len(names)
    if args.classes.strip().lower() != "all":
        want = parse_class_list(args.classes, names)
        empty = [c for c in want if counts[c] == 0]
        if empty:
            raise SystemExit(
                "requested classes have no usable samples: %s\n"
                "  zero single-label samples under --split %s --multilabel %s."
                % (", ".join("%d:%s" % (c, names[c]) for c in empty), args.split, args.multilabel))
        if args.top_k is not None or args.min_samples != 20:
            notes.append("explicit --classes -> --top-k / --min-samples ignored")
        small = [c for c in want if counts[c] < args.min_samples]
        if small:
            notes.append("kept %d class(es) below --min-samples=%d (explicit selection wins): %s"
                         % (len(small), args.min_samples,
                            ", ".join("%s(%d)" % (names[c], counts[c]) for c in small)))
        return want, notes

    keep = [c for c in range(C) if counts[c] >= args.min_samples]
    if not keep:
        raise SystemExit("no class has >= %d usable samples; lower --min-samples" % args.min_samples)
    if args.top_k is not None:
        keep = sorted(sorted(keep, key=lambda c: -counts[c])[:args.top_k])
    return keep, notes

def balanced_subsample(X, y, keep, max_per_class, max_points, seed):
    if not keep or max_per_class < 1 or max_points < len(keep):
        raise ValueError("sample limits must allow at least one point per class")
    k = min(int(max_per_class), int(max_points) // len(keep))
    rng = np.random.default_rng(seed)
    plotted, rows = {}, []
    for c in keep:
        idx = np.nonzero(y == c)[0]
        if idx.size > k:
            idx = rng.choice(idx, size=k, replace=False)
        plotted[c] = int(idx.size)
        rows.append(idx)
    rows = np.sort(np.concatenate(rows))
    return X[rows], y[rows], plotted, k

# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def knn_scores(X, y, k, seed, metric="cosine"):
    counts = np.bincount(y)
    counts = counts[counts > 0]
    if counts.size < 2:
        return None, None, 0
    min_count = int(counts.min())
    n_splits = int(min(5, min_count))
    if n_splits < 2:
        return None, None, 0
    if len(y) > KNN_MAX_SAMPLES:
        from sklearn.model_selection import train_test_split
        sel, _ = train_test_split(np.arange(len(y)), train_size=KNN_MAX_SAMPLES,
                                  stratify=y, random_state=seed)
        X, y = X[sel], y[sel]
        min_count = int(np.unique(y, return_counts=True)[1].min())
        n_splits = min(5, min_count)
        if n_splits < 2:
            return None, None, 0
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = KNeighborsClassifier(n_neighbors=int(min(k, max(1, min_count - 1))),
                               metric=metric, weights="distance")
    pred = cross_val_predict(clf, X, y, cv=cv)
    return accuracy_score(y, pred), f1_score(y, pred, average="macro"), n_splits

def read_auroc(dataset, mode, results_root, tag, feature_kind="cls", *,
               result_file=None, probe_seed=42, budget="melp", return_source=False):
    """Match provenance, never infer a result from a stage filename alone.

    Repeated identical exported records are accepted; conflicting matches fail.
    Explicit files must also contain matching encoder/preprocessing metadata.
    """
    files = [Path(result_file)] if result_file else sorted(Path(results_root).rglob("results.json"))
    matches = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            if result_file:
                raise ValueError("cannot read results file: %s" % path) from exc
            warnings.warn("Skipping unreadable results: %s" % path)
            continue
        enc = data.get("encoder", {})
        if enc.get("encoder_tag") != tag:
            continue
        if mode != "fusion" and enc.get("single_modal_feature") != feature_kind:
            continue
        if data.get("preprocess", {}).get("tag") != PreprocessConfig().tag():
            continue
        for run in data.get("runs", []):
            if (run.get("task") == dataset and run.get("mode") == mode
                    and abs(float(run.get("ratio", -1)) - 1.0) < 1e-9
                    and run.get("seed") == probe_seed and run.get("budget") == budget):
                value = run.get("test_metrics", {}).get("macro_auroc")
                if value is not None and np.isfinite(float(value)):
                    matches.append((float(value) * 100, str(path), run.get("run_id")))
    if len({round(m[0], 10) for m in matches}) > 1:
        raise ValueError("conflicting AUROC records; specify --results-file: %s" % matches)
    if not matches:
        if result_file:
            raise ValueError("explicit results file has no matching encoder/feature/preprocess/run")
        warnings.warn("No matching AUROC: %s %s %s; title will omit AUC" % (dataset, mode, tag))
    value = matches[0][0] if matches else None
    source = {"files": sorted({m[1] for m in matches}), "run_ids": sorted({str(m[2]) for m in matches}),
              "encoder_tag": tag, "probe_seed": probe_seed, "budget": budget,
              "scope": "full test set, all task labels, training ratio 1.0"}
    return (value, source) if return_source else value

# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------

def embed(X, args):
    if len(X) < 3 or not np.isfinite(X).all():
        raise ValueError("embedding needs >=3 finite samples")
    if args.metric == "tsne" and not 0 < args.perplexity < len(X):
        raise ValueError("perplexity must be positive and smaller than sample count")
    Xn = l2(X) if args.normalize == "l2" else np.asarray(X, dtype=np.float64)
    Xn = Xn - Xn.mean(axis=0, keepdims=True)
    if args.metric == "pca":
        return PCA(n_components=2, random_state=args.pca_seed).fit_transform(Xn), "PCA(2)"
    if args.pca_dim < 2:
        raise ValueError("pca-dim must be >=2")
    dim = int(min(args.pca_dim, Xn.shape[1], max(2, Xn.shape[0] - 1)))
    pca = PCA(n_components=dim, random_state=args.pca_seed)
    Xp = pca.fit_transform(Xn)
    if args.pca_whiten == "std":

        Xp = Xp / np.clip(Xp.std(axis=0, keepdims=True), 1e-12, None)
    tsne = TSNE(perplexity=float(args.perplexity), random_state=args.seed, **tsne_kwargs(1000))
    return tsne.fit_transform(Xp), "PCA(%d, variance=%.1f%%, whiten=%s) -> t-SNE" % (
        dim, 100 * pca.explained_variance_ratio_.sum(), args.pca_whiten)

# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def caption(args, meta):
    return ("Features: frozen encoder via linear_probe cache | %s | plotted=%d points, %d classes | "
            "per-class cap=%d | perplexity=%g | %s | normalise=%s | multilabel=%s | split=%s | "
            "seed=%d | legend n = plotted/available | kNN on selected samples (N=%d, capped at 20000)"
            % (meta["run_name"], meta["n_points"], meta["n_classes"], meta["per_class_cap"],
               args.perplexity, meta["reducer"], args.normalize, args.multilabel, args.split,
               args.seed, meta["knn_n"])) + " | kNN distance=%s | sample seed=%d" % (args.knn_metric, args.sample_seed)

def _scatter(ax, Y, y, keep, names, plotted, counts, colors, alpha, size):
    for j, c in enumerate(keep):
        m = y == c
        ax.scatter(Y[m, 0], Y[m, 1], s=size, c=[colors[j]], alpha=alpha, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

def _handles(names, keep, plotted, counts, colors):
    return [Line2D([], [], marker="o", linestyle="", markersize=7,
                   markerfacecolor=colors[j], markeredgecolor="none",
                   label="%s (%d/%d)" % (names[c], plotted[c], int(counts[c])))
            for j, c in enumerate(keep)]

def plot_panel(Y, y, keep, names, plotted, counts, args, meta, out_dir, mode):
    colors = palette(len(keep))
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    _scatter(ax, Y, y, keep, names, plotted, counts, colors, args.alpha, args.point_size)
    title = "%s - %s" % (meta["display"], MODE_LABEL[mode])
    sub = []
    if meta.get("knn_acc") is not None:
        sub.append("kNN acc %.3f" % meta["knn_acc"])
    if meta.get("auroc") is not None:
        sub.append("full-test probe AUROC %.2f" % meta["auroc"])
    if meta.get("knn_f1") is not None:
        sub.append("kNN macro-F1 %.3f" % meta["knn_f1"])
    if sub:
        title += "\n" + "   ".join(sub)
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(handles=_handles(names, keep, plotted, counts, colors),
              loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False,
              ncol=1 if len(keep) <= 22 else 2, handletextpad=0.4, labelspacing=0.35)
    fig.text(0.02, 0.015, caption(args, meta), fontsize=7.2, color="0.35", ha="left")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / ("%s.%s" % (mode, ext)), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

def plot_combined(panels, args, out_dir):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(7.2 * n, 7.4), squeeze=False)
    for ax, p in zip(axes[0], panels):
        colors = palette(len(p["keep"]))
        _scatter(ax, p["Y"], p["y"], p["keep"], p["names"], p["plotted"], p["counts"],
                 colors, args.alpha, args.point_size)
        sub = []
        if p["meta"].get("knn_acc") is not None:
            sub.append("kNN %.3f" % p["meta"]["knn_acc"])
        if p["meta"].get("auroc") is not None:
            sub.append("full-test AUROC %.2f" % p["meta"]["auroc"])
        if p["meta"].get("knn_f1") is not None:
            sub.append("F1 %.3f" % p["meta"]["knn_f1"])
        ax.set_title(MODE_LABEL[p["mode"]] + ("\n" + "   ".join(sub) if sub else ""),
                     fontsize=13, pad=10)
    p0 = panels[0]
    fig.legend(handles=_handles(p0["names"], p0["keep"], p0["plotted"], p0["counts"],
                                palette(len(p0["keep"]))),
               loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=False,
               ncol=1 if len(p0["keep"]) <= 22 else 2, handletextpad=0.4, labelspacing=0.4)
    fig.text(0.012, 0.012, caption(args, p0["meta"]), fontsize=7.2, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / ("combined.%s" % ext), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="t-SNE clustering of frozen lauer features (reads the linear_probe cache)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="run with --list-classes to print the index -> class-name table for a dataset")
    p.add_argument("--dataset", default="ptbxl_rhythm",
                   help='Dataset.')
    p.add_argument("--classes", default="all",
                   help='Classes to display: all, class names, numeric indices, or a comma-separated mixture.')
    p.add_argument("--top-k", type=int, default=None,
                   help='Top k.')
    p.add_argument("--min-samples", type=int, default=20,
                   help='Min samples.')
    p.add_argument("--list-classes", action="store_true",
                   help='List classes.')
    p.add_argument("--mode", default="all", choices=list(MODES) + ["all"],
                   help='Mode.')
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help='Checkpoint.')
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help='Config.')
    p.add_argument("--amp", default=DEFAULT_AMP, choices=["bf16", "fp16", "none"],
                   help='Amp.')
    p.add_argument("--feature-kind", default=DEFAULT_FEATURE_KIND, choices=["cls", "sgl"],
                   help='Feature kind.')
    p.add_argument("--split", default="all", choices=["all", "test", "val", "train"],
                   help='Split.')
    p.add_argument("--multilabel", default="single", choices=["single", "rarest", "drop"],
                   help='Multilabel.')
    p.add_argument("--max-per-class", type=int, default=300,
                   help='Max per class.')
    p.add_argument("--max-points", type=int, default=6000,
                   help='Max points.')
    p.add_argument("--perplexity", type=float, default=30.0,
                   help='Number of neighbors used by t-SNE.')
    p.add_argument("--pca-dim", type=int, default=50,
                   help='Pca dim.')
    p.add_argument("--normalize", default="l2", choices=["l2", "none"],
                   help='Feature normalization applied before dimensionality reduction.')
    p.add_argument("--pca-whiten", default="none", choices=["none", "std"],
                   help='Pca whiten.')
    p.add_argument("--metric", default="tsne", choices=["tsne", "pca"],
                   help='Metric.')
    p.add_argument("--knn", dest="knn", action="store_true", default=True,
                   help='Knn.')
    p.add_argument("--no-knn", dest="knn", action="store_false", help='No knn.')
    p.add_argument("--knn-k", type=int, default=20, help='Knn k.')
    p.add_argument("--seed", type=int, default=42, help='Seed.')
    p.add_argument("--sample-seed", type=int, default=42, help="Fixed sampling/CV seed, independent of t-SNE")
    p.add_argument("--pca-seed", type=int, default=42, help="Fixed PCA seed, independent of t-SNE")
    p.add_argument("--alpha", type=float, default=0.75, help='Alpha.')
    p.add_argument("--point-size", type=float, default=14.0, help='Point size.')
    p.add_argument("--dpi", type=int, default=300, help='Dpi.')
    p.add_argument("--out", default="visualization/t_sne/outputs", help='Out.')
    p.add_argument("--results-root", default=RESULTS_ROOT,
                   help='Results root.')
    p.add_argument("--results-file", default=None, help="Explicit results.json; provenance must match")
    p.add_argument("--probe-seed", type=int, default=42, help="Probe training seed, independent of t-SNE seed")
    p.add_argument("--probe-budget", default="melp")
    p.add_argument("--knn-metric", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--run-name", default=None,
                   help='Run name.')
    return p

def main(argv=None):
    cli = list(sys.argv[1:] if argv is None else argv)
    if "--weights-dir" in cli or any(a.startswith("--weights-dir=") for a in cli):
        from visualization.t_sne.stage_pipeline import main as directory_main
        return directory_main(cli)
    args = build_parser().parse_args(argv)
    np.random.seed(args.seed)

    if args.list_classes:
        keys = DATASET_ORDER if args.dataset == "all" else [args.dataset]
        for i, key in enumerate(keys):
            if i:
                print()
            class_table(key, args.multilabel)
        return 0

    if args.dataset != "all" and args.dataset not in DATASETS:
        raise SystemExit("unknown --dataset %r; choose from %s or all"
                         % (args.dataset, ", ".join(DATASET_ORDER)))
    datasets = DATASET_ORDER if args.dataset == "all" else [args.dataset]
    modes = list(MODES) if args.mode == "all" else [args.mode]

    tag, paths = resolve_cache_tag(args.checkpoint, args.config, args.amp)
    run_name = args.run_name or ("%s_%s" % (Path(args.checkpoint).parent.name,
                                            Path(args.checkpoint).stem))
    out_root = Path(args.out) / run_name
    log("checkpoint  : %s" % args.checkpoint)
    log("config      : %s" % args.config)
    log("encoder tag : %s" % tag)
    log("output      : %s" % out_root)
    log("datasets    : %s" % ", ".join(datasets))
    log("modes       : %s" % ", ".join(modes))

    split_names = ("train", "val", "test") if args.split == "all" else (args.split,)
    summary = ["# t-SNE summary - %s" % run_name, "", "encoder tag: `%s`" % tag, "",
               "split=%s multilabel=%s pca_dim=%d perplexity=%g normalise=%s metric=%s seed=%d "
               "max_per_class=%d max_points=%d" % (args.split, args.multilabel, args.pca_dim,
                                                   args.perplexity, args.normalize, args.metric,
                                                   args.seed, args.max_per_class, args.max_points),
               ""]

    for dataset in datasets:
        ds_dir = out_root / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        panels = []
        reference_ids = reference_labels = None
        summary += ["## %s" % dataset, "",
                    "| mode | classes | plotted | k/class | kNN acc | AUROC | notes |",
                    "|---|---|---|---|---|---|---|"]
        for mode in modes:
            t0 = time.perf_counter()
            X, y, names, counts, dropped, dim, sample_ids = collect(
                dataset, mode, split_names, args.multilabel, args.feature_kind, tag, paths, return_ids=True)
            if reference_ids is None:
                reference_ids, reference_labels = sample_ids.copy(), y.copy()
            elif not (np.array_equal(sample_ids, reference_ids) and np.array_equal(y, reference_labels)):
                raise ValueError("modalities must have identical ordered sample IDs and labels")
            keep, notes = select_classes(args, names, counts)
            mask = np.isin(y, keep)
            Xsel, ysel = X[mask], y[mask]

            knn_acc = knn_f1 = None
            if args.knn:
                knn_acc, knn_f1, _ = knn_scores(l2(Xsel) if args.normalize == "l2"
                                                else Xsel.astype(np.float64), ysel, args.knn_k,
                                                args.sample_seed, metric=args.knn_metric)
            Xs, ys, plotted, k = balanced_subsample(Xsel, ysel, keep, args.max_per_class,
                                                    args.max_points, args.sample_seed)
            Y, reducer = embed(Xs, args)
            auc, auc_source = read_auroc(dataset, mode, args.results_root, tag, args.feature_kind,
                result_file=args.results_file, probe_seed=args.probe_seed,
                budget=args.probe_budget, return_source=True)
            meta = {"run_name": run_name, "display": DATASETS[dataset][1], "n_points": len(ys),
                    "n_classes": len(keep), "reducer": reducer, "knn_acc": knn_acc,
                    "knn_f1": knn_f1, "per_class_cap": k, "knn_n": int(len(ysel)),
                    "auroc": auc, "auc_source": auc_source, "knn_metric": args.knn_metric}
            plotted_rows, _, _, _ = balanced_subsample(np.arange(len(ysel)), ysel, keep,
                args.max_per_class, args.max_points, args.sample_seed)
            (ds_dir / (mode + ".json")).write_text(json.dumps({"args": vars(args), "meta": meta,
                "sample_ids": sample_ids[mask][plotted_rows].tolist(), "labels": ys.tolist(),
                "coordinates": Y.tolist()}, indent=2), encoding="utf-8")
            plot_panel(Y, ys, keep, names, plotted, counts, args, meta, ds_dir, mode)
            panels.append({"Y": Y, "y": ys, "keep": keep, "names": names, "counts": counts,
                           "plotted": plotted, "meta": meta, "mode": mode})
            log("%-18s %-11s plotted=%5d classes=%2d k=%3d knn=%s auroc=%s (%.1fs)"
                % (dataset, mode, len(ys), len(keep), k,
                   "%.3f" % knn_acc if knn_acc is not None else "-",
                   "%.2f" % meta["auroc"] if meta["auroc"] is not None else "-",
                   time.perf_counter() - t0))
            summary.append("| %s | %d/%d | %d | %d | %s | %s | dim=%d; dropped %d multilabel; %s |"
                           % (mode, len(keep), len(names), len(ys), k,
                              "%.3f" % knn_acc if knn_acc is not None else "-",
                              "%.2f" % meta["auroc"] if meta["auroc"] is not None else "-",
                              dim, dropped, "; ".join(notes) or "-"))
        if len(panels) > 1:
            plot_combined(panels, args, ds_dir)
        summary.append("")
        for panel in panels:
            meta = panel["meta"]
            summary.append("- %s: macro-F1=%s; %s; AUC source=%s" % (
                panel["mode"], meta["knn_f1"], meta["reducer"],
                ", ".join(meta["auc_source"]["files"]) or "unavailable"))
        summary.append("")
    (out_root / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    log("summary -> %s" % (out_root / "SUMMARY.md"))
    return 0

if __name__ == "__main__":
    sys.exit(main())
