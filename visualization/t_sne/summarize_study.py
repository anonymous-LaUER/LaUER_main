"""Render study panels and confirm readout effects on the full seven-class test subset."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from visualization.t_sne.tsne_cluster import load_features, PathsConfig, load_public_splits, get_task, l2

OUT = ROOT / "visualization/t_sne/outputs/mechanism_study"
MODES = ["signal_only", "image_only", "fusion"]
LABELS = ["ECG signal", "ECG image", "Fusion"]

def bootstrap_delta(y, a, b, seed=42):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in np.unique(y)]
    values = []
    for _ in range(1000):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        values.append(f1_score(y[idx], a[idx], average="macro") - f1_score(y[idx], b[idx], average="macro"))
    return np.percentile(values, [2.5, 97.5]).tolist()

def main():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    cohort = load_public_splits(get_task("chapman"))
    names = list(cohort.label_names)
    keep = [names.index(n) for n in protocol["class_names"]]
    masks = {s: (cohort.labels[s].sum(1) == 1) & np.isin(cohort.labels[s].argmax(1), keep)
             for s in ["train", "test"]}
    ys = {s: cohort.labels[s][masks[s]].argmax(1) for s in masks}
    ridges = pd.read_csv(OUT / "readouts.csv")
    geometry = pd.read_csv(OUT / "geometry.csv")
    embedding = pd.read_csv(OUT / "embeddings.csv")
    # K-means is an actual unsupervised clustering check with K fixed to seven.
    aris = []
    for r in embedding.itertuples():
        stem = f"{r.stage}__{r.mode}__{r.condition}__p{r.perplexity:g}__s{r.seed}"
        data = np.load(OUT / (stem + ".npz"))
        clusters = KMeans(n_clusters=7, n_init=10, random_state=42).fit_predict(data["coordinates"])
        aris.append(adjusted_rand_score(data["labels"], clusters))
    embedding["kmeans_ari_2d"] = aris
    embedding.to_csv(OUT / "embeddings_with_ari.csv", index=False)
    aggregation = embedding.groupby(["stage", "condition", "mode"])[
        ["silhouette_2d", "purity_2d", "trustworthiness", "kmeans_ari_2d"]].agg(["mean", "std", "min", "max"])
    aggregation.to_csv(OUT / "embedding_summary.csv")
    full, diagnostics, intervals = [], [], []
    for stage in ["sgl_cgl", "sgl_cgl_tgl"]:
        predictions = {}
        source = json.loads((ROOT / "output/Seed42/lauer only clstoken" / stage / "results.json").read_text(encoding="utf-8"))
        for mode in MODES:
            tag = source["encoder"]["encoder_tag"]
            fs = load_features("chapman", mode, tag, "cls", PathsConfig().cache_dir)
            idx = {sid: i for i, sid in enumerate(fs.ids)}
            X = {s: fs.feats[[idx[sid] for sid in cohort.ids[s]]].numpy().astype(np.float64) for s in masks}
            config = ridges[(ridges.stage == stage) & (ridges["mode"] == mode)].iloc[0]
            model = Ridge(alpha=config.ridge_alpha).fit(X["train"], cohort.labels["train"])
            np.savez_compressed(OUT / f"{stage}__{mode}__ridge_model.npz",
                coef=model.coef_, intercept=model.intercept_, alpha=model.alpha,
                labels=np.asarray(names), encoder_tag=tag)
            Z = {s: model.predict(X[s]) for s in X}
            # Measure how much learned score variation survives leading raw PCs.
            reducer = PCA(n_components=min(100, X["train"].shape[1]), svd_solver="randomized", random_state=42).fit(X["train"])
            centered = X["test"] - reducer.mean_
            full_scores = centered @ model.coef_.T
            norms = np.linalg.norm(X["test"], axis=1)
            diag = {"stage": stage, "mode": mode, "norm_cv": float(norms.std()/norms.mean())}
            for dim in [10, 50, 100]:
                V = reducer.components_[:dim]
                approx = (centered @ V.T) @ (V @ model.coef_.T)
                diag[f"score_reconstruction_r2_pca{dim}"] = float(1 - ((full_scores - approx)**2).sum() / (full_scores**2).sum())
            diagnostics.append(diag)
            for kind, features, metric in [("original_cosine", X, "cosine"), ("ridge_scores", Z, "euclidean")]:
                clf = KNeighborsClassifier(n_neighbors=20, metric=metric, weights="distance").fit(features["train"][masks["train"]], ys["train"])
                pred = clf.predict(features["test"][masks["test"]])
                predictions[(mode, kind)] = pred
                full.append({"stage": stage, "mode": mode, "condition": kind,
                    "test_f1": f1_score(ys["test"], pred, average="macro"), "n": len(pred)})
                np.savez_compressed(OUT / f"{stage}__{mode}__{kind}__full_predictions.npz",
                    pred=pred, labels=ys["test"], ids=np.asarray(cohort.ids["test"])[masks["test"]])
            print("confirmed", stage, mode, flush=True)
        for mode in MODES:
            ci = bootstrap_delta(ys["test"], predictions[(mode, "ridge_scores")], predictions[(mode, "original_cosine")])
            intervals.append({"stage": stage, "comparison": mode + ": ridge-original", "ci95": ci})
        for mode in MODES[:2]:
            ci = bootstrap_delta(ys["test"], predictions[("fusion", "ridge_scores")], predictions[(mode, "ridge_scores")])
            intervals.append({"stage": stage, "comparison": "ridge: fusion-" + mode, "ci95": ci})
        # Fixed seed=42 and perplexity=50, selected as a common display setting,
        # not a per-modality best seed. All other settings remain in CSV/NPZ.
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        colors = plt.get_cmap("tab10").colors
        for row, condition in enumerate(["l2_pca50", "ridge_scores"]):
            for col, mode in enumerate(MODES):
                data = np.load(OUT / f"{stage}__{mode}__{condition}__p50__s42.npz")
                ax = axes[row, col]
                for j, c in enumerate(keep):
                    points = data["coordinates"][data["labels"] == c]
                    ax.scatter(points[:, 0], points[:, 1], s=10, color=colors[j], alpha=.75, linewidths=0, label=names[c])
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                f1 = geometry[(geometry.stage == stage) & (geometry["mode"] == mode) & (geometry.condition == condition)].iloc[0].test_f1
                stats = embedding[(embedding.stage == stage) & (embedding["mode"] == mode) & (embedding.condition == condition) & (embedding.seed == 42) & (embedding.perplexity == 50)].iloc[0]
                ax.set_title(f"{LABELS[col]} | held-out kNN F1 {f1:.3f}\n2D silhouette {stats.silhouette_2d:.3f}", fontsize=11)
                if col == 0:
                    ax.set_ylabel("L2 + PCA 50\nUnsupervised geometry" if row == 0 else "38 disease scores\nTrain-label supervised ridge", fontsize=11)
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="outside lower center", ncol=7, frameon=False)
        fig.suptitle(f"{stage} | same {len(protocol['test_ids'])} test records | perplexity 50, seed 42", fontsize=15)
        for ext in ["png", "pdf"]:
            fig.savefig(OUT / f"{stage}__mechanism.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(full).to_csv(OUT / "full_test_confirmation.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "score_geometry.csv", index=False)
    (OUT / "bootstrap.json").write_text(json.dumps(intervals, indent=2), encoding="utf-8")
    print(pd.DataFrame(full).to_string(index=False), flush=True)

if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
