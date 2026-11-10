"""Check downstream split coverage, label alignment, and signal-image lead order."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .cohort import (
    EXPECTED_IMAGE_SIZE,
    coverage_report,
    probe_image_sizes,
    resolve_sample_paths,
)
from .config import PathsConfig, PreprocessConfig, TASK_ORDER, get_task
from .dataset import (
    SignalLoadError,
    load_preprocessed_image,
    load_preprocessed_signal,
    preprocess_signal,
    read_raw_signal,
)
from .splits import SPLITS, load_public_splits

class VerificationError(RuntimeError):
    pass

def _corr(a: np.ndarray, b: np.ndarray, min_points: int = 64) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < min_points:
        return 0.0
    a = a[mask] - a[mask].mean()
    b = b[mask] - b[mask].mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denominator <= 0:
        return 0.0
    return float((a * b).sum() / denominator)

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def _rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(x.size, dtype=np.float64)
        index = 0
        while index < x.size:
            end = index
            while end + 1 < x.size and x[order[end + 1]] == x[order[index]]:
                end += 1
            ranks[order[index:end + 1]] = 0.5 * (index + end) + 1.0
            index = end + 1
        return ranks

    return _corr(_rank(np.asarray(a, dtype=np.float64)),
                 _rank(np.asarray(b, dtype=np.float64)), min_points=8)

def _ptp(signal_row: np.ndarray) -> float:
    return float(np.percentile(signal_row, 99) - np.percentile(signal_row, 1))

def extract_band_traces(image: np.ndarray, n_leads: int = 12,
                        threshold: int = 60, x0: int = 30) -> Optional[np.ndarray]:
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    height = image.shape[0]
    if height % n_leads != 0:
        return None
    band_height = height // n_leads
    darkness = image[:, :, :3].min(axis=2).astype(np.int16)
    rows = np.arange(band_height, dtype=np.float64)[:, None]
    traces = np.full((n_leads, image.shape[1] - x0), np.nan, dtype=np.float64)
    for lead in range(n_leads):
        band = darkness[lead * band_height:(lead + 1) * band_height, :]
        mask = band < threshold
        mask[0, :] = False
        mask[-1, :] = False
        weights = mask.astype(np.float64)
        counts = weights.sum(axis=0)
        centroid = (rows * weights).sum(axis=0) / np.maximum(counts, 1.0)
        centroid[counts == 0] = np.nan
        traces[lead] = centroid[x0:]
    return traces

@dataclass
class FamilyReport:
    task: str
    family: str
    cohort: Dict[str, object]
    coverage: Dict[str, object]
    image_probe: Dict[str, object]
    signal_lead_order: Dict[str, object]
    image_lead_order: Dict[str, object]
    cross_modal: Dict[str, object]
    transformed_lead_order: Dict[str, object]
    label_provenance: Optional[Dict[str, object]]
    failures: List[str] = field(default_factory=list)

def check_code_invariant() -> Dict[str, object]:
    from data.lead_order import (
        LEAD_SWAP_INDICES,
        image_array_to_mimic_order,
        signal_to_mimic_order,
    )

    zeros_probe = np.zeros((12, 4), dtype=np.float32)
    for lead in range(12):
        zeros_probe[lead] = lead
    swapped = signal_to_mimic_order(zeros_probe)[:, 0]
    signal_ok = (
        int(swapped[4]) == 5 and int(swapped[5]) == 4
        and all(int(swapped[k]) == k for k in range(12) if k not in (4, 5))
    )

    image_probe = np.zeros((720, 4, 3), dtype=np.uint8)
    for lead in range(12):
        image_probe[lead * 60:(lead + 1) * 60, :, 0] = lead + 1
    swapped_image = image_array_to_mimic_order(image_probe)[:, 0, 0]
    image_ok = (
        int(swapped_image[240]) == 6 and int(swapped_image[300]) == 5
        and int(swapped_image[0]) == 1 and int(swapped_image[660]) == 12
    )
    return {
        "swap_indices": list(LEAD_SWAP_INDICES),
        "signal_invariant_ok": bool(signal_ok),
        "image_invariant_ok": bool(image_ok),
        "ok": bool(signal_ok and image_ok),
    }

def check_signal_lead_order(cohort, resolved, paths: PathsConfig,
                           preprocess: PreprocessConfig,
                           sample_ids: Sequence[str]) -> Dict[str, object]:
    raw_pre = PreprocessConfig(signal_swap_leads=False)
    swapped_pre = PreprocessConfig(signal_swap_leads=True)
    rows: List[Dict[str, object]] = []
    decisive = 0
    correct = 0
    swapped_ok = 0
    corr_vl_values: List[float] = []
    corr_vf_after_values: List[float] = []
    for sample_id in sample_ids:
        path = resolved[sample_id].signal
        if path is None:
            continue
        try:
            source = read_raw_signal(path, cohort.family, preprocess)
            raw = preprocess_signal(source, raw_pre)
            swapped = preprocess_signal(source, swapped_pre)
        except SignalLoadError:
            continue
        lead_i, lead_ii = raw[0], raw[1]
        a_vl = lead_i - lead_ii / 2.0
        a_vf = lead_ii - lead_i / 2.0
        corr_vl = _corr(raw[4], a_vl)
        corr_vf = _corr(raw[4], a_vf)
        margin = abs(corr_vl - corr_vf)
        is_decisive = margin >= 0.2
        if is_decisive:
            decisive += 1
            if corr_vl > corr_vf:
                correct += 1
        after = _corr(swapped[4], a_vf)
        corr_vl_values.append(corr_vl)
        corr_vf_after_values.append(after)
        if after >= 0.9:
            swapped_ok += 1
        rows.append({
            "sample_id": sample_id,
            "corr_row4_aVL": round(corr_vl, 4),
            "corr_row4_aVF": round(corr_vf, 4),
            "decisive": bool(is_decisive),
            "row4_is_aVL": bool(corr_vl > corr_vf),
            "corr_row4_aVF_after_swap": round(after, 4),
        })
    n = len(rows)
    ratio = (correct / decisive) if decisive else 0.0
    ok = n > 0 and decisive >= max(3, int(0.6 * n)) and ratio == 1.0 and swapped_ok == n
    return {
        "n_sampled": n,
        "n_decisive": decisive,
        "row4_is_aVL_all_decisive": bool(ratio == 1.0 and decisive > 0),
        "n_swapped_matches_aVF": swapped_ok,

        "mean_corr_row4_aVL_raw": round(float(np.mean(corr_vl_values)), 4) if corr_vl_values else 0.0,
        "mean_corr_row4_aVF_after_swap": (
            round(float(np.mean(corr_vf_after_values)), 4) if corr_vf_after_values else 0.0
        ),
        "goldberger_threshold": 0.7,
        "meets_goldberger_threshold": bool(
            corr_vl_values and min(corr_vl_values) >= 0.7
            and corr_vf_after_values and min(corr_vf_after_values) >= 0.7
        ),
        "ok": bool(ok),
        "examples": rows[:5],
    }

def check_image_lead_order(resolved, sample_ids: Sequence[str],
                           n_leads: int = 12) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    decisive = 0
    correct = 0
    limb_ok = 0
    for sample_id in sample_ids:
        path = resolved[sample_id].image
        if path is None:
            continue
        from PIL import Image

        with Image.open(path) as handle:
            image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        traces = extract_band_traces(image, n_leads=n_leads)
        if traces is None:
            continue
        lead_i, lead_ii = traces[0], traces[1]
        lead_iii, lead_avr = traces[2], traces[3]
        band4, band5 = traces[4], traces[5]
        corr_iii = _corr(lead_iii, lead_ii - lead_i)
        corr_avr = _corr(lead_avr, -(lead_i + lead_ii) / 2.0)

        usable = bool(corr_iii >= 0.5 and corr_avr >= 0.7)
        if usable:
            limb_ok += 1
        band4_vl = _corr(band4, lead_i - lead_ii / 2.0)
        band4_vf = _corr(band4, lead_ii - lead_i / 2.0)
        band5_vl = _corr(band5, lead_i - lead_ii / 2.0)
        band5_vf = _corr(band5, lead_ii - lead_i / 2.0)
        margin = abs(band4_vl - band4_vf)
        is_decisive = (usable and margin >= 0.2
                       and np.isfinite(band4_vl) and np.isfinite(band4_vf))
        if is_decisive:
            decisive += 1
            if band4_vl > band4_vf:
                correct += 1
        rows.append({
            "sample_id": sample_id,
            "corr_III_identity": round(corr_iii, 4),
            "corr_aVR_identity": round(corr_avr, 4),
            "band4_aVL": round(band4_vl, 4),
            "band4_aVF": round(band4_vf, 4),
            "band5_aVL": round(band5_vl, 4),
            "band5_aVF": round(band5_vf, 4),
            "usable_trace": usable,
            "band4_is_aVL": bool(band4_vl > band4_vf),
        })
    n = len(rows)
    ratio = (correct / decisive) if decisive else 0.0
    ok = n > 0 and decisive >= max(2, int(0.4 * n)) and ratio == 1.0
    return {
        "n_sampled": n,
        "n_decisive": decisive,
        "n_limb_identity_ok": limb_ok,
        "band4_is_aVL_ratio": round(ratio, 4),
        "ok": bool(ok),
        "examples": [r for r in rows if r["usable_trace"]][:5] or rows[:5],
    }

_RENDER_WINDOW = 5000

def _pool_to_width(signal_row: np.ndarray, width: int, x0: int) -> np.ndarray:
    n = signal_row.shape[0]
    edges = np.linspace(0, n, width + 1).astype(int)
    pooled = np.array([
        signal_row[edges[i]:max(edges[i] + 1, edges[i + 1])].mean()
        for i in range(width)
    ])
    return pooled[x0:]

def check_cross_modal(cohort, resolved, preprocess: PreprocessConfig,
                      sample_ids: Sequence[str]) -> Dict[str, object]:
    from PIL import Image

    render_pre = PreprocessConfig(signal_swap_leads=False)
    per_lead: List[List[float]] = []
    spearman_values: List[float] = []
    n_used = 0
    for sample_id in sample_ids:
        item = resolved[sample_id]
        if item.signal is None or item.image is None:
            continue
        try:
            signal = preprocess_signal(
                read_raw_signal(item.signal, cohort.family, preprocess), render_pre
            )
        except SignalLoadError:
            continue
        with Image.open(item.image) as handle:
            image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        traces = extract_band_traces(image)
        if traces is None:
            continue
        width = traces.shape[1]
        x0 = image.shape[1] - width
        correlations = []
        for lead in range(min(traces.shape[0], signal.shape[0])):
            pooled = _pool_to_width(signal[lead], image.shape[1], x0)
            correlations.append(abs(_corr(traces[lead], -pooled)))
        per_lead.append(correlations)
        n_used += 1

        signal_ptp = np.array([_ptp(signal[k]) for k in range(signal.shape[0])])
        image_spread = np.array([
            float(np.nanmax(traces[k]) - np.nanmin(traces[k]))
            if np.isfinite(traces[k]).any() else 0.0
            for k in range(12)
        ])
        if (signal_ptp > 0).all():
            spearman_values.append(_spearman(image_spread, signal_ptp))

    if not per_lead:
        return {"n_sampled": 0, "lead_corr_mean": 0.0, "ptp_spearman": 0.0, "ok": False}

    matrix = np.asarray(per_lead, dtype=np.float64)
    lead_mean = matrix.mean(axis=0)
    overall = float(lead_mean.mean())
    return {
        "n_sampled": n_used,
        "render_window": _RENDER_WINDOW,
        "lead_corr_mean": round(overall, 4),
        "lead_corr_min": round(float(lead_mean.min()), 4),
        "per_lead_corr": [round(float(v), 4) for v in lead_mean],
        "ptp_spearman": round(float(np.mean(spearman_values)), 4) if spearman_values else 0.0,

        "ok": bool(overall >= 0.5 and float(lead_mean.min()) >= 0.15),
    }

def check_transformed_lead_order(cohort, resolved, preprocess: PreprocessConfig,
                                 sample_ids: Sequence[str]) -> Dict[str, object]:
    from PIL import Image

    effective = PreprocessConfig(
        signal_swap_leads=preprocess.signal_swap_leads,
        image_swap_bands=preprocess.image_swap_bands,
        csn_apply_gain=preprocess.csn_apply_gain,
    )
    per_lead: List[List[float]] = []
    band4_row4: List[float] = []
    band4_row5: List[float] = []
    band5_row5: List[float] = []
    band5_row4: List[float] = []
    n_used = 0
    for sample_id in sample_ids:
        item = resolved[sample_id]
        if item.signal is None or item.image is None:
            continue
        try:
            signal = load_preprocessed_signal(item.signal, cohort.family, effective)
            image = load_preprocessed_image(item.image, effective)
        except SignalLoadError:
            continue
        rgb = np.ascontiguousarray(image.transpose(1, 2, 0))     # [H, W, 3] uint8
        traces = extract_band_traces(rgb)
        if traces is None:
            continue
        width = traces.shape[1]
        x0 = rgb.shape[1] - width
        pooled_rows = [
            _pool_to_width(signal[lead], rgb.shape[1], x0)
            for lead in range(min(traces.shape[0], signal.shape[0]))
        ]
        correlations = [abs(_corr(traces[lead], -pooled_rows[lead]))
                        for lead in range(len(pooled_rows))]
        per_lead.append(correlations)
        band4_row4.append(correlations[4])
        band4_row5.append(abs(_corr(traces[4], -pooled_rows[5])))
        band5_row4.append(abs(_corr(traces[5], -pooled_rows[4])))
        band5_row5.append(correlations[5])
        n_used += 1

    if not per_lead:
        return {"n_sampled": 0, "lead_corr_mean": 0.0, "per_lead_corr": [],
                "band4_vs_row4_avg": 0.0, "band4_vs_row5_avg": 0.0,
                "band5_vs_row5_avg": 0.0, "band5_vs_row4_avg": 0.0,
                "band4_is_row4": False, "band5_is_row5": False, "ok": False}

    matrix = np.asarray(per_lead, dtype=np.float64)
    lead_mean = matrix.mean(axis=0)
    overall = float(lead_mean.mean())
    b4_r4 = float(np.mean(band4_row4))
    b4_r5 = float(np.mean(band4_row5))
    b5_r5 = float(np.mean(band5_row5))
    b5_r4 = float(np.mean(band5_row4))
    a_vf_ok = b4_r4 > b4_r5 and b4_r4 >= 0.5
    a_vl_ok = b5_r5 > b5_r4 and b5_r5 >= 0.5
    return {
        "n_sampled": n_used,
        "lead_corr_mean": round(overall, 4),
        "per_lead_corr": [round(float(v), 4) for v in lead_mean],
        "band4_vs_row4_avg": round(b4_r4, 4),
        "band4_vs_row5_avg": round(b4_r5, 4),
        "band5_vs_row5_avg": round(b5_r5, 4),
        "band5_vs_row4_avg": round(b5_r4, 4),
        "band4_is_row4": bool(a_vf_ok),
        "band5_is_row5": bool(a_vl_ok),
        "ok": bool(a_vf_ok and a_vl_ok and overall >= 0.5),
    }

def check_label_provenance(task_key: str, paths: PathsConfig) -> Dict[str, object]:
    from .label_audit import audit_task

    report = audit_task(task_key, paths, splits=("train", "val", "test"))
    failures: List[str] = []
    gate: Dict[str, object] = {}
    for comparison in report["comparisons"]:
        name = comparison["name"]
        exact = float(comparison["exact_match_rate"])
        raw_only_total = sum(item["raw_only"]
                             for item in comparison["per_class"].values())
        gate[name] = {
            "exact_match_rate": exact,
            "dead_classes_in_raw": comparison["dead_classes_in_raw"],
            "raw_only_total": raw_only_total,
        }
        if "unstripped" in name:
            continue
        if exact != 1.0 or comparison["n_common"] != report["n_csv_labels"]:
            failures.append(
                f"{task_key}: source annotation agreement must be exactly 1.0 on all rows; "
                f"got {exact:.6f}, common={comparison['n_common']}, expected={report['n_csv_labels']}"
            )
    report["gate"] = gate
    report["failures"] = failures
    report["ok"] = not failures
    return report

def verify_task(task_key: str,
                paths: PathsConfig,
                preprocess: PreprocessConfig,
                n_signal: int = 12,
                n_image: int = 12,
                strict_counts: bool = True,
                verbose: bool = True,
                label_order: str = "name",
                label_audit: bool = True) -> FamilyReport:
    task = get_task(task_key)
    failures: List[str] = []

    cohort = load_public_splits(task, strict_counts=strict_counts,
                                label_order=label_order)
    resolved = resolve_sample_paths(cohort, paths, check_exists=False)
    coverage = coverage_report(cohort, paths)

    for split in SPLITS:
        if coverage[split]["missing_signal"] or coverage[split]["missing_image"]:
            failures.append(
                f"{task_key}/{split}: missing_signal={coverage[split]['missing_signal']} "
                f"missing_image={coverage[split]['missing_image']}"
            )

    min_positive = {
        split: int(cohort.labels[split].sum(axis=0).min()) for split in SPLITS
    }
    for split, value in min_positive.items():
        if value <= 0:
            dead = [
                name for index, name in enumerate(cohort.label_names)
                if cohort.labels[split][:, index].sum() == 0
            ]
            failures.append(f"{task_key}/{split}: dead classes {dead}")

    sample_ids: List[str] = []
    per_split = max(1, n_image // 3)
    for split in SPLITS:
        sample_ids.extend(cohort.ids[split][:per_split])
    sample_ids = sample_ids[:max(n_image, n_signal)]
    available = [s for s in sample_ids if resolved[s].image is not None]

    image_probe = probe_image_sizes(resolved, available)
    if not image_probe["matches_expected"]:
        failures.append(
            f"{task_key}: unexpected image sizes {image_probe['sizes']} "
            f"(expected {EXPECTED_IMAGE_SIZE[0]}x{EXPECTED_IMAGE_SIZE[1]})"
        )
    if not image_probe["fully_opaque"]:
        failures.append(
            f"{task_key}: images are not fully opaque {image_probe['alpha_ranges']} — "
            "convert('RGB') in the loader would composite alpha differently from MEETI"
        )

    signal_order = check_signal_lead_order(
        cohort, resolved, paths, preprocess, sample_ids[:n_signal]
    )
    if not signal_order["ok"]:
        failures.append(f"{task_key}: signal lead order check failed {signal_order}")

    image_order = check_image_lead_order(resolved, available[:n_image],
                                         n_leads=preprocess.n_leads)
    if not image_order["ok"]:
        failures.append(f"{task_key}: image lead order check failed {image_order}")

    cross = check_cross_modal(cohort, resolved, preprocess, available[:n_image])
    if not cross["ok"]:
        failures.append(f"{task_key}: image<->signal band alignment check failed {cross}")

    transformed = check_transformed_lead_order(
        cohort, resolved, preprocess, available[:n_image]
    )
    if not transformed["ok"]:
        failures.append(
            f"{task_key}: post-transform multimodal lead alignment failed {transformed}"
        )

    label_provenance = (
        check_label_provenance(task_key, paths) if label_audit else None
    )
    if label_provenance is not None and not label_provenance["ok"]:
        failures.extend(f"{task_key}: {item}" for item in label_provenance["failures"])

    report = FamilyReport(
        task=task_key,
        family=task.family,
        cohort={
            "counts": cohort.counts(),
            "expected": list(task.expected_counts),
            "num_classes": cohort.num_classes,
            "fingerprint": cohort.fingerprint,
            "csv_sha256": cohort.csv_sha256,
            "label_order": cohort.label_order,
            "label_permutation_nontrivial": {
                k: v for k, v in cohort.label_permutation.items()
                if v != list(range(len(v)))
            },
            "min_positive_per_class": min_positive,
            "warnings": cohort.warnings,
        },
        coverage=coverage,
        image_probe=image_probe,
        signal_lead_order=signal_order,
        image_lead_order=image_order,
        cross_modal=cross,
        transformed_lead_order=transformed,
        label_provenance=label_provenance,
        failures=failures,
    )
    return report

def run_verification(tasks: Optional[Sequence[str]] = None,
                     paths: Optional[PathsConfig] = None,
                     preprocess: Optional[PreprocessConfig] = None,
                     n_signal: int = 12,
                     n_image: int = 12,
                     strict_counts: bool = True,
                     verbose: bool = True,
                     label_order: str = "name",
                     label_audit: bool = True) -> Dict[str, object]:
    paths = paths or PathsConfig.from_env()
    preprocess = preprocess or PreprocessConfig()
    task_keys = list(tasks) if tasks else list(TASK_ORDER)

    invariant = check_code_invariant()
    reports: Dict[str, object] = {}
    all_failures: List[str] = []
    if not invariant["ok"]:
        all_failures.append(f"lead-order code invariant broken: {invariant}")

    for task_key in task_keys:
        report = verify_task(task_key, paths, preprocess, n_signal=n_signal,
                             n_image=n_image, strict_counts=strict_counts,
                             verbose=verbose, label_order=label_order,
                             label_audit=label_audit)
        reports[task_key] = report
        all_failures.extend(report.failures)
        if verbose:
            status = "OK  " if not report.failures else "FAIL"
            print(f"[{status}] {task_key:20s} "
                  f"n={report.cohort['counts']} "
                  f"signal_lead_ok={report.signal_lead_order['ok']} "
                  f"image_lead_ok={report.image_lead_order['ok']} "
                  f"band_align={report.cross_modal['lead_corr_mean']} "
                  f"post_swap={report.transformed_lead_order.get('lead_corr_mean', 0.0)} "
                  f"labels={'ok' if report.label_provenance is None or report.label_provenance['ok'] else 'FAIL'}")
            for warning in report.cohort["warnings"]:
                print(f"        warn: {warning}")
            for failure in report.failures:
                print(f"        FAIL: {failure}")

    return {
        "invariant": invariant,
        "tasks": reports,
        "failures": all_failures,
        "ok": not all_failures,
    }

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="linear_probe pre-flight verification (cohort/coverage/lead order)"
    )
    parser.add_argument("--tasks", nargs="*", default=None,
                        help=f"tasks to check (default: all of {list(TASK_ORDER)})")
    parser.add_argument("--n-signal", type=int, default=12,
                        help="records sampled per task for the signal lead-order check")
    parser.add_argument("--n-image", type=int, default=12,
                        help="images sampled per task for the image lead-order check")
    parser.add_argument("--no-strict-counts", action="store_true",
                        help="do not enforce the published row counts")
    parser.add_argument("--no-label-audit", action="store_true",
                        help="skip the raw-vs-CSV label provenance cross-check")
    parser.add_argument("--label-order", choices=["name", "positional"], default="name")
    parser.add_argument("--json-out", type=str, default="",
                        help="write the full report JSON to this path")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    result = run_verification(tasks=args.tasks, n_signal=args.n_signal,
                              n_image=args.n_image,
                              strict_counts=not args.no_strict_counts,
                              label_order=args.label_order,
                              label_audit=not args.no_label_audit)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[verify] report written to {path}")

    print()
    if result["ok"]:
        print("=== verification PASSED ===")
        return 0
    print(f"=== verification FAILED ({len(result['failures'])} issue(s)) ===")
    for failure in result["failures"]:
        print(f"  - {failure}")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
