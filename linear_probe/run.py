"""Train and evaluate downstream linear probes on frozen LaUER representations."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import __version__
from .cohort import coverage_report, merge_family_cohorts
from .config import (
    BUDGETS,
    MODES,
    RATIOS,
    TASK_ORDER,
    PathsConfig,
    PreprocessConfig,
    ProbeConfig,
    get_budget,
    get_task,
)
from .features import EncoderBundle, build_bundle, encoder_tag, extract_family_features
from .metrics import format_metrics
from .probe import evaluate_features, train_linear_probe
from .head_io import head_identity, save_head
RATIO_ORDER = RATIOS
from .splits import SPLITS, load_public_splits, subsample_train

DEVIATIONS: Tuple[str, ...] = (
    "Validation selection uses logits for classwise AUROC; cross-class softmax changes ranking. "
    "",
    "The linear head is trained in FP32.",
    "Autograd anomaly detection is disabled.",
    "Label columns are aligned by name; PTB-XL subclass validation/test orders differ from training. "
    "See splits.py for the explicit label-order policy.",
)

@dataclass
class RunRecord:

    run_id: str
    task: str
    display_name: str
    family: str
    mode: str
    ratio: float
    budget: str
    subset_scheme: str
    seed: int
    in_dim: int
    num_classes: int
    n_train: int
    n_val: int
    n_test: int
    val_metrics: Dict[str, object]
    test_metrics: Dict[str, object]
    epochs_run: int
    best_epoch: int
    stopped_early: bool
    degraded: bool
    history: List[Dict[str, object]] = field(default_factory=list)
    classifier_path: str = ""

class ProbeRunner:

    def __init__(self, paths: PathsConfig, preprocess: PreprocessConfig,
                 probe: ProbeConfig, use_cache: bool = True,
                 verify_first: bool = True, verbose: bool = True,
                 planned_modes: Sequence[str] = (),
                 label_order: str = "name"):
        self.paths = paths
        self.preprocess = preprocess
        self.probe = probe
        self.use_cache = use_cache
        self.verify_first = verify_first
        self.verbose = verbose
        self.device = probe.device if torch.cuda.is_available() else "cpu"
        self._cohorts: Dict[str, object] = {}
        self._features: Dict[Tuple[str, str], object] = {}
        self._family_cohorts: Dict[str, object] = {}
        self._bundle: Optional[EncoderBundle] = None
        self.feature_tag = encoder_tag(probe, paths)
        if label_order not in ("name", "positional"):
            raise ValueError(f"label_order must be name|positional, got {label_order!r}")

        self.label_order = label_order

        self._planned_modes: Tuple[str, ...] = tuple(planned_modes) or ()
        if self._planned_modes:
            unknown = [m for m in self._planned_modes if m not in MODES]
            if unknown:
                raise ValueError(f"unknown planned modes {unknown}")

    def cohort(self, task_key: str):
        if task_key not in self._cohorts:
            cohort = load_public_splits(get_task(task_key),
                                        label_order=self.label_order)
            if self.verify_first:
                report = coverage_report(cohort, self.paths)
                for split in SPLITS:
                    item = report[split]
                    if item["missing_signal"] or item["missing_image"]:
                        raise RuntimeError(
                            f"{task_key}/{split}: missing_signal="
                            f"{item['missing_signal']} missing_image="
                            f"{item['missing_image']} — run linear_probe.verify"
                        )
            self._cohorts[task_key] = cohort
            if self.verbose:
                for warning in cohort.warnings:
                    print(f"[run] warn: {warning}")
        return self._cohorts[task_key]

    def family_cohort(self, family: str):
        if family not in self._family_cohorts:
            task_keys = [t for t in TASK_ORDER if get_task(t).family == family]
            self._family_cohorts[family] = merge_family_cohorts(
                [self.cohort(t) for t in task_keys]
            )
        return self._family_cohorts[family]

    def ensure_features(self, family: str, modes: Sequence[str]):

        prep_tag = self.preprocess.tag()

        def _key(mode: str):
            return (family, mode, prep_tag)

        wanted = [m for m in modes if _key(m) not in self._features]
        if not wanted:
            return
        cohort = self.family_cohort(family)
        if self.verbose:
            print(f"[run] family cohort {family}: "
                  f"{ {s: len(cohort.ids[s]) for s in SPLITS} }, "
                  f"union={len(cohort.all_ids())}")
        if self._bundle is None:
            build_modes = set(self._planned_modes) | set(wanted)
            self._bundle = build_bundle(self.probe, build_modes, self.device)
        elif not set(wanted) <= set(self._bundle.modes):

            previous = set(self._bundle.modes)
            print(f"[run] rebuilding encoder bundle for "
                  f"{sorted(set(wanted) - previous)}")
            self._bundle.close()
            self._bundle = build_bundle(self.probe, previous | set(wanted), self.device)

        sets = extract_family_features(
            cohort, self.paths, self.preprocess, self.probe, wanted,
            self.device, use_cache=self.use_cache, bundle=self._bundle,
        )
        for mode, item in sets.items():
            self._features[_key(mode)] = item
            if self.verbose:
                print(f"[run] features {family}/{mode}: "
                      f"{tuple(item.feats.shape)} (dim={item.dim}) prep={prep_tag}")

    def features(self, family: str, mode: str):
        self.ensure_features(family, [mode])
        return self._features[(family, mode, self.preprocess.tag())]

    def prepare(self, tasks: Sequence[str], modes: Sequence[str]) -> None:
        families: List[str] = []
        for task_key in tasks:
            family = get_task(task_key).family
            if family not in families:
                families.append(family)
        for family in families:
            self.ensure_features(family, list(modes))

    def close(self) -> None:
        if self._bundle is not None:
            self._bundle.close()
            self._bundle = None

    def run_single(self, task_key: str, mode: str, ratio: float, budget_name: str,
                   seed: int) -> RunRecord:
        task = get_task(task_key)
        cohort = self.cohort(task_key)
        budget = get_budget(budget_name)
        feature_set = self.features(task.family, mode)

        train_ids = subsample_train(cohort.ids["train"], ratio,
                                    scheme=budget.subset_scheme, seed=42)
        val_ids = list(cohort.ids["val"])
        test_ids = list(cohort.ids["test"])

        def _tensor(labels: np.ndarray, split: str, ids: List[str]) -> torch.Tensor:
            rows = [cohort.row_index[split][i] for i in ids]
            return torch.from_numpy(labels[rows].copy())

        train_features = feature_set.rows_for(train_ids)
        train_labels = _tensor(cohort.labels["train"], "train", train_ids)
        val_features = feature_set.rows_for(val_ids)
        val_labels = _tensor(cohort.labels["val"], "val", val_ids)
        test_features = feature_set.rows_for(test_ids)
        test_labels = _tensor(cohort.labels["test"], "test", test_ids)
        label_names = cohort.label_names

        if self.verbose:
            print(f"[run] {task_key} | {mode} | ratio={ratio:g} | {budget_name} | "
                  f"seed={seed} | train={len(train_ids)} val={len(val_ids)} "
                  f"test={len(test_ids)} dim={train_features.shape[1]} "
                  f"| signal={self.preprocess.n_leads}x{self.preprocess.signal_length}")

        outcome = train_linear_probe(
            train_features, train_labels, val_features, val_labels, label_names,
            budget, device=self.device, seed=seed, verbose=self.verbose,
        )
        test_metrics, _ = evaluate_features(
            outcome.classifier, test_features, test_labels, label_names, self.device
        )
        identity = head_identity(self.feature_tag, self.preprocess, cohort, task_key, mode,
                                 self.probe.single_modal_feature, ratio, budget_name, seed)
        classifier_path = save_head(self.paths.cache_dir, identity, outcome, test_metrics,
                                    selection=budget.select)
        record = RunRecord(
            run_id=(f"{task_key}__{mode}__r{ratio:g}__{budget_name}__seed{seed}"),
            task=task_key,
            display_name=task.display_name,
            family=task.family,
            mode=mode,
            ratio=ratio,
            budget=budget_name,
            subset_scheme=budget.subset_scheme,
            seed=seed,
            in_dim=int(train_features.shape[1]),
            num_classes=cohort.num_classes,
            n_train=len(train_ids),
            n_val=len(val_ids),
            n_test=len(test_ids),
            val_metrics=outcome.val_metrics,
            test_metrics=test_metrics,
            epochs_run=outcome.epochs_run,
            best_epoch=outcome.best_epoch,
            stopped_early=outcome.stopped_early,
            degraded=outcome.degraded,
            history=outcome.history,
            classifier_path=str(classifier_path),
        )
        if self.verbose:
            print(f"[run] test: {format_metrics(test_metrics)}")
        return record

    def run_matrix(self, tasks: Sequence[str], modes: Sequence[str],
                   ratios: Sequence[float], budgets: Sequence[str],
                   seeds: Sequence[int]) -> List[RunRecord]:
        records: List[RunRecord] = []
        total = len(tasks) * len(modes) * len(ratios) * len(budgets) * len(seeds)
        done = 0
        started = time.time()
        for task_key in tasks:
            for mode in modes:
                for ratio in ratios:
                    for budget_name in budgets:
                        for seed in seeds:
                            done += 1
                            header = (f"[{done}/{total}] {task_key} {mode} "
                                      f"r={ratio:g} {budget_name} seed={seed}")
                            print("\n" + "=" * 72)
                            print(header)
                            print("=" * 72)
                            try:
                                records.append(self.run_single(
                                    task_key, mode, ratio, budget_name, seed
                                ))
                            except Exception as error:
                                print(f"[run] FAILED {header}: {error}")
                            elapsed = time.time() - started
                            print(f"[run] elapsed {elapsed/60:.1f} min, "
                                  f"eta {elapsed/done*(total-done)/60:.1f} min")
        return records

def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def environment_info() -> Dict[str, object]:
    info: Dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "linear_probe": __version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
    return info

def save_records(records: Sequence[RunRecord], paths: PathsConfig,
                 preprocess: PreprocessConfig, probe: ProbeConfig,
                 cohorts: Dict[str, object], tag: str,
                 encoder_tag_value: str) -> Path:
    out_dir = Path(paths.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "environment": environment_info(),
        "deviations": list(DEVIATIONS),
        "encoder": {
            "checkpoint": probe.encoder_checkpoint,
            "config_yaml": probe.config_yaml,
            "encoder_tag": encoder_tag_value,
            "single_modal_feature": probe.single_modal_feature,
            "amp": probe.amp,
        },
        "preprocess": {
            "tag": preprocess.tag(),
            **{k: v for k, v in asdict(preprocess).items()},
        },
        "label_source": "melp_public_split_csv",
        "cohorts": {
            task_key: {
                "counts": cohort.counts(),
                "fingerprint": cohort.fingerprint,
                "num_classes": cohort.num_classes,
                "label_names": cohort.label_names,
                "csv_sha256": cohort.csv_sha256,
                "label_order": cohort.label_order,
                "label_source": "melp_public_split_csv",
                "label_permutation_nontrivial": {
                    k: v for k, v in cohort.label_permutation.items()
                    if v != list(range(len(v)))
                },
                "warnings": cohort.warnings,
            }
            for task_key, cohort in cohorts.items()
        },
    }
    json_path = out_dir / "results.json"

    merged_runs: "Dict[str, Dict[str, object]]" = {}
    if json_path.is_file():
        previous: Optional[Dict[str, object]] = None
        try:
            previous = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as error:
            print(f"[run] WARNING: existing {json_path} unreadable ({error}); "
                  "it will be replaced")
        if previous is not None:
            previous_encoder = str(
                (previous.get("encoder") or {}).get("encoder_tag") or ""
            )
            previous_preprocess = str(
                (previous.get("preprocess") or {}).get("tag") or ""
            )
            if previous_encoder and previous_encoder != encoder_tag_value:
                raise RuntimeError(
                    f"{json_path} was produced with encoder {previous_encoder!r}, "
                    f"this run uses {encoder_tag_value!r} — refusing to merge "
                    "different encoders; use a different --tag"
                )
            if previous_preprocess and previous_preprocess != preprocess.tag():
                raise RuntimeError(
                    f"{json_path} was produced with preprocess {previous_preprocess!r}, "
                    f"this run uses {preprocess.tag()!r} — refusing to merge "
                    "different preprocess calibers; use a different --tag"
                )
            for item in previous.get("runs", []):
                run_id = str(item.get("run_id"))
                if run_id:
                    merged_runs[run_id] = item
            for task_key, cohort in previous.get("cohorts", {}).items():
                payload["cohorts"].setdefault(task_key, cohort)
    for record in records:
        merged_runs[record.run_id] = _jsonable(asdict(record))
    payload["runs"] = list(merged_runs.values())
    payload["aggregates"] = aggregate_serialized(payload["runs"])

    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(json_path)
    return json_path

def aggregate_serialized(runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    buckets: Dict[str, List[Dict[str, object]]] = {}
    for run in runs:
        key = (f"{run['task']}|{run['mode']}|{run['ratio']:g}|{run['budget']}"
               if isinstance(run["ratio"], float) else
               f"{run['task']}|{run['mode']}|{float(run['ratio']):g}|{run['budget']}")
        buckets.setdefault(key, []).append(run)
    out: Dict[str, object] = {}
    for key, items in buckets.items():
        aurocs = [float(item["test_metrics"]["macro_auroc"]) for item in items]
        f1s = [float(item["test_metrics"]["macro_f1"]) for item in items]
        out[key] = {
            "n_seeds": len(items),
            "seeds": [item["seed"] for item in items],
            "auroc_mean": float(np.mean(aurocs)) * 100.0,
            "auroc_std": (float(np.std(aurocs, ddof=1)) * 100.0
                          if len(aurocs) > 1 else 0.0),
            "f1_mean": float(np.mean(f1s)) * 100.0,
            "n_classes_scored": int(items[0]["test_metrics"]["n_classes_scored"]),
            "n_classes_total": int(items[0]["test_metrics"]["n_classes_total"]),
            "n_train": items[0]["n_train"],
        }
    return out

def aggregate(records: Sequence[RunRecord]) -> Dict[str, object]:
    buckets: Dict[str, List[RunRecord]] = {}
    for record in records:
        key = f"{record.task}|{record.mode}|{record.ratio:g}|{record.budget}"
        buckets.setdefault(key, []).append(record)
    out: Dict[str, object] = {}
    for key, items in buckets.items():
        aurocs = [float(r.test_metrics["macro_auroc"]) for r in items]
        f1s = [float(r.test_metrics["macro_f1"]) for r in items]
        out[key] = {
            "n_seeds": len(items),
            "seeds": [r.seed for r in items],
            "auroc_mean": float(np.mean(aurocs)) * 100.0,
            "auroc_std": float(np.std(aurocs, ddof=1)) * 100.0 if len(aurocs) > 1 else 0.0,
            "f1_mean": float(np.mean(f1s)) * 100.0,
            "n_classes_scored": int(items[0].test_metrics["n_classes_scored"]),
            "n_classes_total": int(items[0].test_metrics["n_classes_total"]),
            "n_train": items[0].n_train,
        }
    return out

def print_summary(records: Sequence[RunRecord]) -> None:
    by_key: Dict[Tuple[str, str, str, float], List[float]] = {}
    for record in records:
        key = (record.task, record.mode, record.budget, record.ratio)
        by_key.setdefault(key, []).append(float(record.test_metrics["macro_auroc"]))

    for mode in MODES:
        rows = [k for k in by_key if k[1] == mode]
        if not rows:
            continue
        print()
        print("=" * 88)
        print(f"  MODE = {mode}   (macro AUROC %, test split, best validation checkpoint)")
        print("=" * 88)
        header = (f"  {'task':20s} {'budget':7s} " +
                  " ".join(f"{int(r*100):>8d}%" for r in RATIO_ORDER) +
                  "")
        print(header)
        print("  " + "-" * 84)
        for task_key in TASK_ORDER:
            for budget_name in sorted(BUDGETS):
                cells = []
                missing = False
                for ratio in RATIO_ORDER:
                    values = by_key.get((task_key, mode, budget_name, ratio))
                    if not values:
                        cells.append(f"{'-':>9s}")
                        missing = True
                    else:
                        mean = statistics.fmean(values) * 100.0
                        cells.append(f"{mean:>9.2f}")
                print(f"  {task_key:20s} {budget_name:7s} " + " ".join(cells)
                      )

        print("  " + "-" * 84)
        for budget_name in sorted(BUDGETS):
            cells = []
            for ratio in RATIO_ORDER:
                values = [value for key, bucket in by_key.items()
                          if key[1] == mode and key[2] == budget_name
                          and key[3] == ratio for value in bucket]
                cells.append(f"{statistics.fmean(values)*100.0:>9.2f}" if values
                             else f"{'-':>9s}")
            print(f"  {'MEAN':20s} {budget_name:7s} " + " ".join(cells))

# CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="linear_probe: MELP-aligned downstream linear probing"
    )
    parser.add_argument("--task", default="ptbxl_super_class",
                        choices=list(TASK_ORDER))
    parser.add_argument("--mode", default="signal_only",
                        help="signal_only | image_only | fusion | all")
    parser.add_argument("--ratio", type=float, default=1.0, choices=list(RATIOS))
    parser.add_argument("--budget", default="melp", choices=sorted(BUDGETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])

    parser.add_argument("--matrix", action="store_true",
                        help="run the full matrix instead of a single configuration")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_ORDER))
    parser.add_argument("--modes", nargs="+", default=["signal_only"],
                        help="list of modes, or 'all'")
    parser.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    parser.add_argument("--budgets", nargs="+", default=["melp"],
                        help="list of budgets (melp merl)")

    parser.add_argument("--encoder-checkpoint", type=str,
                        default="weights/lauer_sgl_cgl_tgl.pt")
    parser.add_argument("--config", type=str,
                        default="configs/pretrain_SGL+CGL+TGL.yaml")
    parser.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--extract-batch-size", type=int, default=0)
    parser.add_argument("--single-modal-feature", choices=["sgl", "cls"], default="cls",
                        help="Unimodal features: cls = raw encoder CLS "
                             "(default); "
                             "sgl = normalized SGL projected embeddings")

    parser.add_argument("--label-order", choices=["name", "positional"], default="name")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore and do not write the feature cache")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the coverage pre-check (not recommended)")
    parser.add_argument("--tag", default="", help="output subdirectory name")
    parser.add_argument("--dry-run", action="store_true")
    return parser

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    paths = PathsConfig.from_env()

    preprocess = PreprocessConfig()
    probe = ProbeConfig(
        seeds=tuple(args.seeds),
        device=args.device,
        amp=args.amp,
        num_workers=args.num_workers,
        extract_batch_size=args.extract_batch_size,
        single_modal_feature=args.single_modal_feature,
        encoder_checkpoint=args.encoder_checkpoint,
        config_yaml=args.config,
    )

    if args.matrix:
        modes = list(MODES) if args.modes == ["all"] else list(args.modes)
    else:
        modes = list(MODES) if args.mode == "all" else [args.mode]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise SystemExit(
            f"unknown modes {unknown}; expected {list(MODES)} or 'all'"
        )

    if args.matrix:
        jobs = [(t, m, r, b, s) for t in args.tasks for m in modes
                for r in args.ratios for b in args.budgets for s in args.seeds]
    else:
        jobs = [(args.task, m, args.ratio, args.budget, s)
                for m in modes for s in args.seeds]

    print(f"[run] {len(jobs)} probe(s); "
          f"encoder={args.encoder_checkpoint}; label_order={args.label_order}")
    for job in jobs:
        print(f"       {job}")
    if args.dry_run:
        return 0

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.label_order == "positional":
        print("[run] WARNING: label_order='positional' reproduces MELP/MERL's "
              "positional label read (ptbxl_sub_class val/test columns are permuted).")
    runner = ProbeRunner(paths, preprocess, probe, use_cache=not args.no_cache,
                         verify_first=not args.no_verify, planned_modes=modes,
                         label_order=args.label_order)
    records: List[RunRecord] = []
    try:
        if args.matrix:

            runner.prepare(args.tasks, modes)
            records = runner.run_matrix(args.tasks, modes, args.ratios,
                                        args.budgets, args.seeds)
        else:
            for task_key, mode, ratio, budget, seed in jobs:
                records.append(runner.run_single(task_key, mode, ratio, budget, seed))
    finally:
        runner.close()

    cohorts: Dict[str, object] = {}
    for key in sorted({job[0] for job in jobs}):
        try:
            cohorts[key] = runner.cohort(key)
        except Exception as error:
            print(f"[run] WARNING: cohort metadata for {key} unavailable: {error}")
    json_path = save_records(records, paths, preprocess, probe, cohorts, tag,
                             runner.feature_tag)
    print(f"\n[run] {len(records)} record(s) written to {json_path}")
    print_summary(records)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
