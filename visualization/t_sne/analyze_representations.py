"""Paired exploratory study: geometry, supervised readout and t-SNE sensitivity.

Uses existing caches only. PCA and ridge readout fit train only; ridge alpha is
selected using full multilabel validation AUROC. Test sweeps are exploratory,
not a fresh independent benchmark after hyperparameter selection.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import f1_score, silhouette_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from visualization.t_sne.tsne_cluster import (
    MODES, PathsConfig, get_task, load_public_splits, load_features,
    balanced_subsample, l2, tsne_kwargs, read_auroc,
)

def auc(y, z):
    valid = (y.sum(0) > 0) & (y.sum(0) < len(y))
    return float(roc_auc_score(y[:, valid], z[:, valid], average="macro"))

def neighborhood(X, y):
    clf = NearestNeighbors(n_neighbors=min(16, len(y)), metric="euclidean")
    clf.fit(X)
    # Explicit X includes self: remove self by index rather than assuming first
    # neighbour is self (duplicate representations can tie).
    inds = clf.kneighbors(X, return_distance=False)
    scores = [np.mean(y[row[row != i][:15]] == y[i]) for i, row in enumerate(inds)]
    return float(np.mean([np.mean(np.asarray(scores)[y == c]) for c in np.unique(y)]))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="visualization/t_sne/outputs/mechanism_study")
    p.add_argument("--seeds", default="17,42,66")
    p.add_argument("--perplexities", default="15,50")
    p.add_argument("--max-per-class", type=int, default=100)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cohort = load_public_splits(get_task("chapman"))
    names = list(cohort.label_names)
    # Exactly the seven classes in the user's current plot; not chosen by outcome.
    keep = [names.index(n) for n in ["SA", "ALS", "APB", "AF", "SVT", "TWC", "ST"]]
    single = {s: (cohort.labels[s].sum(1) == 1) & np.isin(cohort.labels[s].argmax(1), keep)
              for s in ["train", "val", "test"]}
    ys = {s: cohort.labels[s][single[s]].argmax(1) for s in single}
    rows, y, _, _ = balanced_subsample(np.arange(len(ys["test"])), ys["test"], keep,
        args.max_per_class, 6000, 20260915)
    test_ids = np.asarray(cohort.ids["test"])[single["test"]][rows]
    (out / "protocol.json").write_text(json.dumps({
        "args": vars(args), "dataset": "chapman", "class_names": [names[i] for i in keep],
        "test_ids": test_ids.tolist(), "test_labels": y.tolist(), "sample_seed": 20260915,
        "pca_fit": "all training samples", "ridge_fit": "all training samples, all 38 labels",
        "ridge_selection": "validation macro AUROC, alpha in [1,100,10000]",
        "note": "Exploratory test sensitivity sweep. No claim of independent parameter-selection validation.",
    }, indent=2), encoding="utf-8")
    geometry, embeddings, readouts, per_class, norms = [], [], [], [], []
    result_root = ROOT / "output/Seed42/lauer only clstoken"
    for stage in ["sgl_cgl", "sgl_cgl_tgl"]:
        result_file = result_root / stage / "results.json"
        source = json.loads(result_file.read_text(encoding="utf-8"))
        tag = source["encoder"]["encoder_tag"]
        for mode in MODES:
            print("Loading", stage, mode, flush=True)
            fs = load_features("chapman", mode, tag, "cls", PathsConfig().cache_dir)
            lengths = np.linalg.norm(fs.feats.numpy().astype(np.float64), axis=1)
            norms.append({"stage": stage, "mode": mode, "norm_mean": float(lengths.mean()),
                          "norm_cv": float(lengths.std() / lengths.mean()),
                          "norm_min": float(lengths.min()), "norm_max": float(lengths.max())})
            (out / "norms.json").write_text(json.dumps(norms, indent=2), encoding="utf-8")
            index = {sid: i for i, sid in enumerate(fs.ids)}
            X = {s: fs.feats[[index[sid] for sid in cohort.ids[s]]].numpy().astype(np.float64)
                 for s in single}
            original_auc = read_auroc("chapman", mode, str(result_root), tag, result_file=result_file)
            best_auc, ridge = -1, None
            for alpha in [1., 100., 10000.]:
                model = Ridge(alpha=alpha).fit(X["train"], cohort.labels["train"])
                score = auc(cohort.labels["val"], model.predict(X["val"]))
                if score > best_auc:
                    best_auc, ridge = score, model
            Z = {s: ridge.predict(X[s]) for s in X}
            readouts.append({"stage": stage, "mode": mode, "original_probe_auc": original_auc / 100,
                "ridge_alpha": ridge.alpha, "ridge_val_auc": best_auc,
                "ridge_test_auc": auc(cohort.labels["test"], Z["test"])})
            pd.DataFrame(readouts).to_csv(out / "readouts.csv", index=False)
            for condition in ["raw_pca50", "l2_pca50", "l2_pca10", "l2_pca100", "l2_white50", "ridge_scores"]:
                if condition == "ridge_scores":
                    transformed, variance = Z, 1.
                else:
                    base = {s: (X[s] if condition.startswith("raw") else l2(X[s])) for s in X}
                    dim = 100 if condition.endswith("100") else 10 if condition.endswith("10") else 50
                    reducer = PCA(n_components=dim, svd_solver="randomized", random_state=42,
                                  whiten="white" in condition).fit(base["train"])
                    transformed = {s: reducer.transform(base[s]) for s in base}
                    variance = float(reducer.explained_variance_ratio_.sum())
                train = transformed["train"][single["train"]]
                test = transformed["test"][single["test"]][rows]
                val = transformed["val"][single["val"]]
                clf = KNeighborsClassifier(n_neighbors=20, weights="distance", metric="euclidean").fit(train, ys["train"])
                pred = clf.predict(test)
                val_pred = clf.predict(val)
                high_f1 = f1_score(y, pred, average="macro")
                # Original cosine and Euclidean baselines, before PCA/readout.
                if condition == "raw_pca50":
                    for metric in ["cosine", "euclidean"]:
                        raw_clf = KNeighborsClassifier(n_neighbors=20, weights="distance", metric=metric).fit(
                            X["train"][single["train"]], ys["train"])
                        raw_pred = raw_clf.predict(X["test"][single["test"]][rows])
                        geometry.append({"stage": stage, "mode": mode, "condition": "original_" + metric,
                            "test_f1": f1_score(y, raw_pred, average="macro"), "variance": 1.})
                geometry.append({"stage": stage, "mode": mode, "condition": condition,
                    "test_f1": high_f1, "val_f1": f1_score(ys["val"], val_pred, average="macro"),
                    "variance": variance, "silhouette": silhouette_score(test, y),
                    "neighbor_purity": neighborhood(test, y)})
                for c, score in zip(keep, f1_score(y, pred, labels=keep, average=None)):
                    per_class.append({"stage": stage, "mode": mode, "condition": condition,
                                      "class": names[c], "f1": score, "n": int((y == c).sum())})
                pd.DataFrame(geometry).to_csv(out / "geometry.csv", index=False)
                pd.DataFrame(per_class).to_csv(out / "per_class.csv", index=False)
                for perplexity in map(float, args.perplexities.split(",")):
                    for seed in map(int, args.seeds.split(",")):
                        tsne = TSNE(perplexity=perplexity, random_state=seed, **tsne_kwargs(1000))
                        Y = tsne.fit_transform(test)
                        record = {"stage": stage, "mode": mode, "condition": condition,
                            "perplexity": perplexity, "seed": seed,
                            "silhouette_2d": silhouette_score(Y, y),
                            "purity_2d": neighborhood(Y, y),
                            "trustworthiness": trustworthiness(test, Y, n_neighbors=15),
                            "kl": float(tsne.kl_divergence_)}
                        embeddings.append(record)
                        stem = f"{stage}__{mode}__{condition}__p{perplexity:g}__s{seed}"
                        np.savez_compressed(out / (stem + ".npz"), coordinates=Y, labels=y, ids=test_ids)
                        pd.DataFrame(embeddings).to_csv(out / "embeddings.csv", index=False)
                print(stage, mode, condition, "F1=%.4f variance=%.3f" % (high_f1, variance), flush=True)
    print("Completed", out, flush=True)

if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
