"""Three stages x six tasks x raw/BCE/ridge with paired samples and metrics."""
import argparse
import csv
import json
import pickle
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from linear_probe.config import TASK_ORDER, PathsConfig, PreprocessConfig, ProbeConfig, get_task
from linear_probe.features import encoder_tag, feature_cache_path, load_feature_set
from linear_probe.head_io import head_identity, head_path, load_head
from linear_probe.metrics import evaluate_scores
from linear_probe.probe import predict_logits
from linear_probe.run import ProbeRunner, save_records
from linear_probe.splits import load_public_splits
from visualization.t_sne.tsne_cluster import l2, balanced_subsample, knn_scores, palette, tsne_kwargs

STAGES = {"sgl": "pretrain_SGL.yaml", "sgl_cgl": "pretrain_SGL+CGL.yaml",
          "sgl_cgl_tgl": "pretrain_SGL+CGL+TGL.yaml"}
MODES = ("signal_only", "image_only", "fusion")
SPACE_LABELS = {"raw": "CLS / fusion CLS + L2 + PCA50", "bce": "BCE linear probe logits",
                "ridge": "Ridge disease scores"}
MODE_LABELS = {"signal_only": "ECG signal", "image_only": "ECG image", "fusion": "Fusion"}

def modes_for(stage):
    return MODES[:2] if stage == "sgl" else MODES

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def existing_results(root, identity):
    """Only full provenance-bearing result files; task exports lack identity."""
    matches = []
    for path in sorted(Path(root).rglob("results.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        enc = data.get("encoder", {})
        if (enc.get("encoder_tag") != identity["encoder_tag"] or
                data.get("preprocess", {}).get("tag") != identity["preprocess_tag"] or
                enc.get("single_modal_feature") != "cls"):
            continue
        if any(r.get("task") == identity["task"] and r.get("mode") == identity["mode"]
               and r.get("ratio") == identity["ratio"] and r.get("seed") == identity["seed"]
               and r.get("budget") == identity["budget"] for r in data.get("runs", [])):
            matches.append(path.resolve())
    return matches

def make_jobs(args, paths, cohorts):
    jobs = []
    for stage, config_name in STAGES.items():
        checkpoint = Path(args.weights_dir).resolve() / (stage + ".pt")
        config = Path(args.config_dir).resolve() / config_name
        if not checkpoint.is_file() or not config.is_file():
            raise ValueError("missing checkpoint/config: %s / %s" % (checkpoint, config))
        probe = ProbeConfig(encoder_checkpoint=str(checkpoint), config_yaml=str(config),
                            amp=args.amp, single_modal_feature="cls", seeds=(args.probe_seed,))
        tag = encoder_tag(probe, paths)
        # File names route jobs; inspect checkpoint metadata/keys to reject a renamed wrong stage.
        from training.stage_io import artifact_source_mode
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        state = artifact.get("state_dict", artifact.get("model_state_dict", artifact))
        actual = artifact_source_mode(artifact, state)
        if actual.replace("+", "_") != stage:
            raise ValueError("checkpoint stage mismatch: %s reports %s" % (checkpoint, actual))
        del artifact, state
        for task, cohort in cohorts.items():
            for mode in modes_for(stage):
                identity = head_identity(tag, PreprocessConfig(), cohort, task, mode, "cls",
                                         args.ratio, args.probe_budget, args.probe_seed)
                jobs.append(dict(stage=stage, task=task, mode=mode, probe=probe, identity=identity,
                                 head=head_path(paths.cache_dir, identity)))
    return jobs

def preflight(jobs, args, paths):
    missing = []
    for job in jobs:
        try:
            load_head(job["head"], job["identity"])
        except (OSError, ValueError, RuntimeError, KeyError, EOFError, pickle.UnpicklingError) as error:
            item = {k: job[k] for k in ("stage", "task", "mode")}
            item.update(head=str(job["head"]), reason=str(error),
                        overwrite_results=[str(p) for p in existing_results(args.results_root, job["identity"])])
            missing.append((job, item))
    return missing

def approve_missing(missing, args, input_fn=None):
    if not missing:
        return True
    print("\nMissing or invalid BCE classifier heads: %d" % len(missing), flush=True)
    for _, item in missing:
        print("  {stage}/{task}/{mode}".format(**item))
        for path in item["overwrite_results"]:
            print("    Matching records will be replaced in: ", path)
    print("Rerunning replaces records and classifier heads for the listed run IDs.", flush=True)
    if args.missing_heads == "rerun":
        return True  # explicit non-interactive user opt-in
    if args.missing_heads == "error" or (input_fn is None and not sys.stdin.isatty()):
        return False
    answer = (input_fn or input)("Type RERUN to replace the listed runs; any other input cancels: ")
    return answer.strip() == "RERUN"

def rerun_missing(missing, args, paths, cohorts):
    for stage in STAGES:
        selected = [(j, item) for j, item in missing if j["stage"] == stage]
        if not selected:
            continue
        runner = ProbeRunner(paths, PreprocessConfig(), selected[0][0]["probe"],
                             planned_modes=modes_for(stage))
        try:
            for job, item in selected:
                record = runner.run_single(job["task"], job["mode"], args.ratio,
                                           args.probe_budget, args.probe_seed)
                destinations = [Path(p) for p in item["overwrite_results"]]
                if not destinations:
                    destinations = [Path(paths.output_dir) / Path(args.weights_dir).name / stage / "results.json"]
                for dest in destinations:
                    save_records([record], replace(paths, output_dir=str(dest.parent.parent)),
                                 PreprocessConfig(), job["probe"], {job["task"]: cohorts[job["task"]]},
                                 dest.parent.name, job["identity"]["encoder_tag"])
        finally:
            runner.close()

def samples(cohort, args):
    ids = np.concatenate([np.asarray(cohort.ids[s]) for s in ("train", "val", "test")])
    labels = np.concatenate([cohort.labels[s] for s in ("train", "val", "test")])
    mask = labels.sum(1) == 1
    ids, y = ids[mask], labels[mask].argmax(1)
    counts = np.bincount(y, minlength=len(cohort.label_names))
    keep = np.flatnonzero(counts >= args.min_samples).tolist()
    if len(keep) < 2:
        raise ValueError("need at least two eligible single-label classes")
    rows, selected_y, plotted, cap = balanced_subsample(np.arange(len(y)), y, keep,
        args.max_per_class, args.max_points, args.sample_seed)
    if args.perplexity >= len(rows):
        raise ValueError("perplexity must be smaller than plotted sample count")
    return ids[rows].tolist(), selected_y, keep, counts, plotted

def spaces(X, fs, cohort, head, job, directory):
    reducer = PCA(n_components=min(50, X.shape[1], len(X)-1), random_state=42, svd_solver="full")
    raw = reducer.fit_transform(l2(X))
    bce = predict_logits(head, torch.as_tensor(X, dtype=torch.float32), "cpu").numpy()
    ridge_path = directory / "ridge.npz"
    provenance = dict(identity=job["identity"], alphas=[1., 100., 10000.], selection="val_macro_auroc")
    key = json.dumps(provenance, sort_keys=True)
    coef = intercept = alpha = None
    if ridge_path.is_file():
        with np.load(ridge_path, allow_pickle=False) as data:
            if str(data["provenance"]) == key:
                coef, intercept, alpha = data["coef"], data["intercept"], float(data["alpha"])
    if coef is None:
        train = fs.rows_for(cohort.ids["train"]).numpy().astype(np.float64)
        val = fs.rows_for(cohort.ids["val"]).numpy().astype(np.float64)
        best, fitted = -np.inf, None
        for a in provenance["alphas"]:
            model = Ridge(alpha=a).fit(train, cohort.labels["train"])
            score = evaluate_scores(cohort.labels["val"], model.predict(val), cohort.label_names,
                                    score_space="logit")["macro_auroc"]
            if np.isfinite(score) and score > best:
                best, fitted = score, model
        if fitted is None:
            raise ValueError("ridge validation AUC undefined")
        coef, intercept, alpha = fitted.coef_, fitted.intercept_, fitted.alpha
        ridge_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ridge_path, coef=coef, intercept=intercept, alpha=alpha, provenance=key)
    return {"raw": raw, "bce": bce, "ridge": X @ coef.T + intercept}, {
        "pca_variance": float(reducer.explained_variance_ratio_.sum()),
        "pca_solver": "full", "ridge_alpha": alpha, "ridge_path": str(ridge_path)}

def embed_and_score(X, y, args, auc):
    if not np.isfinite(X).all():
        raise ValueError("non-finite features")
    acc, f1, folds = knn_scores(X, y, 20, args.sample_seed, metric="euclidean")
    if folds < 2:
        raise ValueError("selected classes need at least two samples for kNN CV")
    Y = TSNE(perplexity=args.perplexity, random_state=args.seed, **tsne_kwargs(1000)).fit_transform(X)
    metrics = dict(linear_probe_auroc=float(auc), silhouette_2d=float(silhouette_score(Y, y)),
                   knn_accuracy=float(acc), knn_macro_f1=float(f1))
    return Y, metrics

def plot(panels, cohort, selection, args, stage, space, directory):
    _, y, keep, counts, plotted = selection
    colors = palette(len(keep))
    def render(items, name):
        cols = len(items)
        legend_cols = min(3 if cols == 1 else 6, len(keep))
        legend_rows = int(np.ceil(len(keep)/legend_cols))
        fig, axes = plt.subplots(1, cols, figsize=(6*cols, 6.7+legend_rows*.28), squeeze=False)
        for ax, panel in zip(axes[0], items):
            for i, c in enumerate(keep):
                Y = panel["Y"][y == c]
                ax.scatter(Y[:, 0], Y[:, 1], s=10, alpha=.72, color=colors[i], linewidths=0,
                    label="%s (%d/%d)" % (cohort.label_names[c], plotted[c], counts[c]))
            m = panel["metrics"]
            ax.set_title(MODE_LABELS[panel["mode"]], fontsize=13, pad=53)
            ax.text(.5, 1.025, "BCE probe AUROC %.2f%% | 2D silhouette %.3f\nkNN accuracy %.3f | kNN macro-F1 %.3f" % (
                100*m["linear_probe_auroc"], m["silhouette_2d"], m["knn_accuracy"], m["knn_macro_f1"]),
                transform=ax.transAxes, ha="center", fontsize=9, linespacing=1.7)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, [textwrap.fill(label, 25) for label in labels], loc="lower center",
                   bbox_to_anchor=(.5, .075), ncol=legend_cols, frameon=False, fontsize=8)
        title = "%s | %s\n%s" % (stage, get_task(args.current_task).display_name, SPACE_LABELS[space])
        fig.suptitle(title, fontsize=12 if cols == 1 else 14, y=.98)
        footer = "All-splits single-label subset | N=%d | perplexity=%g | seed=%d\n" % (len(y), args.perplexity, args.seed)
        footer += "AUROC: full-test BCE reference | kNN: pre-t-SNE Euclidean CV\nDiagnostic only; includes readout training samples"
        fig.text(.5, .013, footer, ha="center", fontsize=7, color=".35")
        fig.subplots_adjust(left=.035, right=.98, top=.79, bottom=.17+legend_rows*.035, wspace=.12)
        directory.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(directory/(name+"."+ext), dpi=200)
        plt.close(fig)
    for panel in panels:
        render([panel], {"signal_only": "signal", "image_only": "image", "fusion": "fusion"}[panel["mode"]])
    render(panels, "combined")

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights-dir", required=True)
    p.add_argument("--config-dir", required=True)
    p.add_argument("--dataset", choices=["all"]+list(TASK_ORDER), default="all")
    p.add_argument("--out", default="visualization/t_sne/outputs")
    p.add_argument("--results-root", default="output")
    p.add_argument("--missing-heads", choices=["ask", "error", "rerun"], default="ask")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--ratio", type=float, choices=[1.0], default=1.0)
    p.add_argument("--probe-budget", choices=["melp"], default="melp")
    p.add_argument("--probe-seed", type=int, default=42)
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--perplexity", type=float, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--max-per-class", type=int, default=300)
    p.add_argument("--max-points", type=int, default=6000)
    p.add_argument("--min-samples", type=int, default=20)
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.perplexity <= 0 or args.min_samples < 2:
        raise ValueError("perplexity must be positive; min-samples must be >=2")
    paths = PathsConfig.from_env()
    tasks = list(TASK_ORDER) if args.dataset == "all" else [args.dataset]
    cohorts = {t: load_public_splits(get_task(t)) for t in tasks}
    selections = {t: samples(c, args) for t, c in cohorts.items()}
    jobs = make_jobs(args, paths, cohorts)
    root = Path(args.out).resolve()/Path(args.weights_dir).resolve().name
    missing = preflight(jobs, args, paths)
    dump(root/"preflight.json", {"jobs": len(jobs), "missing_heads": [item for _, item in missing]})
    if args.preflight_only:
        print("Preflight: %d jobs, %d missing heads; %s" % (len(jobs), len(missing), root/"preflight.json"))
        return 2 if missing else 0
    if not approve_missing(missing, args):
        print("Stopped without rerunning training. Details: ", root/"preflight.json")
        return 2
    if missing:
        rerun_missing(missing, args, paths, cohorts)
    with threadpool_limits(limits=4):
        for stage in STAGES:
            for task, cohort in cohorts.items():
                selection = selections[task]
                selected_ids, y, keep, counts, plotted = selection
                directory = root/stage/task
                panels = {s: [] for s in SPACE_LABELS}
                metadata, records = [], []
                for job in [j for j in jobs if j["stage"] == stage and j["task"] == task]:
                    mode = job["mode"]
                    path = feature_cache_path(paths.cache_dir, get_task(task).family, mode,
                                              job["identity"]["encoder_tag"], PreprocessConfig().tag(), "cls")
                    fs = load_feature_set(path)
                    if fs is None:
                        raise ValueError("feature cache missing; run linear_probe first: %s" % path)
                    if len(set(fs.ids)) != len(fs.ids):
                        raise ValueError("duplicate cached feature IDs")
                    X = fs.rows_for(selected_ids).numpy().astype(np.float64)
                    head, payload = load_head(job["head"], job["identity"], X.shape[1])
                    auc = payload["test_metrics"]["macro_auroc"]
                    if auc is None or not np.isfinite(auc):
                        raise ValueError("BCE probe test AUC unavailable")
                    transformed, diag = spaces(X, fs, cohort, head, job, directory/"ridge"/mode)
                    metadata.append(dict(mode=mode, head=str(job["head"]), identity=job["identity"], diagnostics=diag))
                    for space, features in transformed.items():
                        Y, metrics = embed_and_score(features, y, args, auc)
                        panels[space].append(dict(mode=mode, Y=Y, metrics=metrics))
                        records.append(dict(space=space, mode=mode, **metrics))
                        dest = directory/space
                        dest.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(dest/(mode+".npz"), coordinates=Y, labels=y,
                                            sample_ids=np.asarray(selected_ids))
                        print(stage, task, mode, space, metrics, flush=True)
                args.current_task = task
                for space in SPACE_LABELS:
                    plot(panels[space], cohort, selection, args, stage, space, directory/space)
                with (directory/"metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                    writer.writeheader(); writer.writerows(records)
                dump(directory/"manifest.json", dict(args=vars(args), stage=stage, task=task,
                    ids=selected_ids, labels=y.tolist(), class_names=list(cohort.label_names),
                    sources=metadata, metrics=records, sklearn=sklearn.__version__,
                    protocol="all-splits, single-label; per-space Euclidean kNN CV before t-SNE; not held-out readout performance"))
    return 0
