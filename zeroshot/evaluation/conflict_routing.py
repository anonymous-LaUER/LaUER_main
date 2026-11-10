"""Conflict-routed CGL inference for split-branch lauer.

The module is deliberately numpy-only.  Raw neural-network scores are produced
once, then every routing and fusion ablation can be reproduced without another
GPU forward pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Tuple

import numpy as np

SCALE_PROFILES: Mapping[str, Tuple[str, str]] = {
    # profile -> (SGL scaler, CGL-margin scaler)
    "raw": ("none", "none"),
    "class_z": ("class_z", "class_z"),
    "class_robust": ("class_robust", "class_robust"),
    "global_z": ("global_z", "global_z"),
    "global_robust": ("global_robust", "global_robust"),
    # Preserve the binary head's zero-margin decision boundary.
    "z_margin_zero": ("class_z", "class_std_zero"),
    "robust_margin_zero": ("class_robust", "class_mad_zero"),
}

CGL_SCORE_KEYS: Mapping[str, Tuple[str, str]] = {
    "margin": ("concept_logit", "concept_logit"),
    "match_logit": ("concept_logit", "concept_logit"),
    "probability": ("concept_logit", "concept_logit"),
}

def _finite_matrix(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must have shape [N,C], got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return result

@dataclass(frozen=True)
class ScoreScaler:
    """Unsupervised affine score scaler fitted without target labels."""

    method: str
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        method: str,
        epsilon: float = 1e-6,
    ) -> "ScoreScaler":
        matrix = _finite_matrix(values, "scaler values")
        valid_methods = {
            "none",
            "class_z",
            "class_robust",
            "global_z",
            "global_robust",
            "class_std_zero",
            "class_mad_zero",
        }
        if method not in valid_methods:
            raise ValueError(
                f"Unknown scaler {method!r}; expected {sorted(valid_methods)}"
            )

        if method == "none":
            center = np.zeros((1, matrix.shape[1]), dtype=np.float64)
            scale = np.ones((1, matrix.shape[1]), dtype=np.float64)
        elif method in {"class_z", "class_std_zero"}:
            center = matrix.mean(axis=0, keepdims=True)
            scale = matrix.std(axis=0, keepdims=True)
            if method.endswith("_zero"):
                center = np.zeros_like(center)
        elif method in {"class_robust", "class_mad_zero"}:
            median = np.median(matrix, axis=0, keepdims=True)
            mad = np.median(np.abs(matrix - median), axis=0, keepdims=True)
            center = median
            scale = 1.4826 * mad
            fallback = matrix.std(axis=0, keepdims=True)
            scale = np.where(scale > epsilon, scale, fallback)
            if method.endswith("_zero"):
                center = np.zeros_like(center)
        elif method == "global_z":
            center = np.asarray([[matrix.mean()]])
            scale = np.asarray([[matrix.std()]])
        else:
            median = float(np.median(matrix))
            mad = float(np.median(np.abs(matrix - median)))
            center = np.asarray([[median]])
            scale = np.asarray([[1.4826 * mad]])
            if scale.item() <= epsilon:
                scale = np.asarray([[matrix.std()]])

        scale = np.where(np.isfinite(scale) & (scale > epsilon), scale, 1.0)
        return cls(method=method, center=center, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = _finite_matrix(values, "scores")
        if self.center.shape[1] not in {1, matrix.shape[1]}:
            raise ValueError(
                "Scaler class count does not match score class count: "
                f"{self.center.shape[1]} vs {matrix.shape[1]}"
            )
        result = (matrix - self.center) / self.scale
        if not np.isfinite(result).all():
            raise ValueError("Scaled scores contain NaN or Inf")
        return result

    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "center": self.center.reshape(-1).tolist(),
            "scale": self.scale.reshape(-1).tolist(),
        }

@dataclass(frozen=True)
class TargetCalibration:
    """Four score-family scalers fitted on one unlabeled validation target."""

    scale_profile: str
    cgl_score_kind: str
    ecg: ScoreScaler
    image: ScoreScaler
    branch1: ScoreScaler
    branch2: ScoreScaler

    @classmethod
    def fit(
        cls,
        raw_scores: Mapping[str, np.ndarray],
        scale_profile: str,
        cgl_score_kind: str = "margin",
    ) -> "TargetCalibration":
        if scale_profile not in SCALE_PROFILES:
            raise ValueError(
                f"Unknown scale profile {scale_profile!r}; expected "
                f"{sorted(SCALE_PROFILES)}"
            )
        sgl_method, cgl_method = SCALE_PROFILES[scale_profile]
        if cgl_score_kind not in CGL_SCORE_KEYS:
            raise ValueError(
                f"Unknown CGL score kind {cgl_score_kind!r}; expected "
                f"{sorted(CGL_SCORE_KEYS)}"
            )
        branch1_key, branch2_key = CGL_SCORE_KEYS[cgl_score_kind]
        return cls(
            scale_profile=scale_profile,
            cgl_score_kind=cgl_score_kind,
            ecg=ScoreScaler.fit(raw_scores["ecg_sgl"], sgl_method),
            image=ScoreScaler.fit(
                raw_scores["image_sgl"], sgl_method
            ),
            branch1=ScoreScaler.fit(
                raw_scores[branch1_key], cgl_method
            ),
            branch2=ScoreScaler.fit(
                raw_scores[branch2_key], cgl_method
            ),
        )

    def transform(
        self, raw_scores: Mapping[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        branch1_key, branch2_key = CGL_SCORE_KEYS[self.cgl_score_kind]
        return {
            "ecg": self.ecg.transform(raw_scores["ecg_sgl"]),
            "image": self.image.transform(raw_scores["image_sgl"]),
            "branch1": self.branch1.transform(
                raw_scores[branch1_key]
            ),
            "branch2": self.branch2.transform(
                raw_scores[branch2_key]
            ),
        }

    def to_dict(self) -> Dict:
        return {
            "scale_profile": self.scale_profile,
            "cgl_score_kind": self.cgl_score_kind,
            "ecg": self.ecg.to_dict(),
            "image": self.image.to_dict(),
            "branch1": self.branch1.to_dict(),
            "branch2": self.branch2.to_dict(),
        }

@dataclass(frozen=True)
class RoutingConfig:
    """Hyperparameters for conflict detection, branch routing, and fusion."""

    scale_profile: str = "class_robust"
    cgl_score_kind: str = "margin"
    branch_policy: str = "specialized"
    gate_mode: str = "hybrid"
    fusion_mode: str = "interpolate"
    confidence_threshold: float = 0.5
    gap_threshold: float = 1.0
    gate_width: float = 0.5
    alpha: float = 0.5
    uncertain_weight: float = 0.0
    ecg_weight: float = 0.5

    def validate(self) -> None:
        if self.scale_profile not in SCALE_PROFILES:
            raise ValueError(f"Unknown scale profile {self.scale_profile}")
        if self.cgl_score_kind not in CGL_SCORE_KEYS:
            raise ValueError(
                f"Unknown CGL score kind {self.cgl_score_kind}"
            )
        if self.branch_policy not in {
            "specialized",
            "reversed",
            "average",
            "branch1_only",
            "branch2_only",
        }:
            raise ValueError(f"Unknown branch policy {self.branch_policy}")
        if self.gate_mode not in {"hard", "gap", "hybrid"}:
            raise ValueError(f"Unknown gate mode {self.gate_mode}")
        if self.fusion_mode not in {"interpolate", "residual"}:
            raise ValueError(f"Unknown fusion mode {self.fusion_mode}")
        if self.confidence_threshold < 0:
            raise ValueError("confidence_threshold must be non-negative")
        if self.gap_threshold < 0:
            raise ValueError("gap_threshold must be non-negative")
        if self.gate_width <= 0:
            raise ValueError("gate_width must be positive")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0,1]")
        if not 0 <= self.uncertain_weight <= 1:
            raise ValueError("uncertain_weight must be in [0,1]")
        if not 0 <= self.ecg_weight <= 1:
            raise ValueError("ecg_weight must be in [0,1]")

    def to_dict(self) -> Dict:
        return asdict(self)

def _ramp(values: np.ndarray, threshold: float, width: float) -> np.ndarray:
    """Continuous gate that is exactly zero below ``threshold``."""

    return np.clip((values - threshold) / width, 0.0, 1.0)

def route_conflicts(
    scaled_scores: Mapping[str, np.ndarray],
    config: RoutingConfig,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Route SGL conflicts to their matching query-guided CGL branch.

    ECG-high/image-low pairs use branch 1, which was trained with ECG-text
    hard negatives.  Image-high/ECG-low pairs use branch 2, trained with
    image-text hard negatives.  The branch input in the neural forward pass is
    still the candidate prompt tokens plus the positive signal-image fusion;
    only the scalar decision score is routed here.
    """

    config.validate()
    ecg = _finite_matrix(scaled_scores["ecg"], "scaled ECG-SGL")
    image = _finite_matrix(scaled_scores["image"], "scaled Image-SGL")
    branch1 = _finite_matrix(
        scaled_scores["branch1"], "scaled branch1 margin"
    )
    branch2 = _finite_matrix(
        scaled_scores["branch2"], "scaled branch2 margin"
    )
    if not (ecg.shape == image.shape == branch1.shape == branch2.shape):
        raise ValueError("All routed score matrices must have the same shape")

    threshold = config.confidence_threshold
    gap = ecg - image
    absolute_gap = np.abs(gap)
    maximum = np.maximum(ecg, image)
    minimum = np.minimum(ecg, image)

    consensus_positive = (ecg >= threshold) & (image >= threshold)
    consensus_negative = (ecg <= -threshold) & (image <= -threshold)
    ecg_conflict = (ecg >= threshold) & (image <= -threshold)
    image_conflict = (image >= threshold) & (ecg <= -threshold)
    common_uncertain = (np.abs(ecg) < threshold) & (
        np.abs(image) < threshold
    )

    if config.gate_mode == "hard":
        conflict_gate = (ecg_conflict | image_conflict).astype(np.float64)
    elif config.gate_mode == "gap":
        conflict_gate = _ramp(
            absolute_gap, config.gap_threshold, config.gate_width
        )
    else:
        gap_gate = _ramp(
            absolute_gap, config.gap_threshold, config.gate_width
        )
        high_gate = _ramp(
            maximum, config.confidence_threshold, config.gate_width
        )
        low_gate = _ramp(
            -minimum, config.confidence_threshold, config.gate_width
        )
        conflict_gate = gap_gate * high_gate * low_gate

    # High-high and low-low are trusted SGL consensus by construction.
    consensus = consensus_positive | consensus_negative
    conflict_gate = np.where(consensus, 0.0, conflict_gate)
    if config.uncertain_weight > 0:
        conflict_gate = np.maximum(
            conflict_gate,
            config.uncertain_weight * common_uncertain.astype(np.float64),
        )

    base = (
        config.ecg_weight * ecg
        + (1.0 - config.ecg_weight) * image
    )
    if config.branch_policy == "specialized":
        routed_branch = np.where(gap >= 0, branch1, branch2)
    elif config.branch_policy == "reversed":
        routed_branch = np.where(gap >= 0, branch2, branch1)
    elif config.branch_policy == "average":
        routed_branch = 0.5 * (branch1 + branch2)
    elif config.branch_policy == "branch1_only":
        routed_branch = branch1
    else:
        routed_branch = branch2
    if config.fusion_mode == "interpolate":
        final = base + config.alpha * conflict_gate * (
            routed_branch - base
        )
    else:
        final = base + config.alpha * conflict_gate * routed_branch

    diagnostics = {
        "base_score": base,
        "routed_branch_score": routed_branch,
        "gate": conflict_gate,
        "gap": gap,
        "consensus_positive": consensus_positive,
        "consensus_negative": consensus_negative,
        "ecg_conflict": ecg_conflict,
        "image_conflict": image_conflict,
        "common_uncertain": common_uncertain,
        "other": ~(
            consensus_positive
            | consensus_negative
            | ecg_conflict
            | image_conflict
            | common_uncertain
        ),
    }
    if not np.isfinite(final).all():
        raise ValueError("Conflict-routed scores contain NaN or Inf")
    return final, diagnostics

def summarize_routing(
    diagnostics: Mapping[str, np.ndarray],
) -> Dict[str, float | int]:
    """Compact counts and gate statistics for experiment logs."""

    gate = np.asarray(diagnostics["gate"], dtype=np.float64)
    total = int(gate.size)
    result: Dict[str, float | int] = {
        "pair_count": total,
        "gate_nonzero_count": int((gate > 0).sum()),
        "gate_nonzero_fraction": float((gate > 0).mean()),
        "gate_full_count": int((gate >= 1).sum()),
        "gate_mean": float(gate.mean()),
    }
    for name in (
        "consensus_positive",
        "consensus_negative",
        "ecg_conflict",
        "image_conflict",
        "common_uncertain",
        "other",
    ):
        mask = np.asarray(diagnostics[name], dtype=bool)
        count = int(mask.sum())
        result[f"{name}_count"] = count
        result[f"{name}_fraction"] = count / total if total else 0.0
    return result
