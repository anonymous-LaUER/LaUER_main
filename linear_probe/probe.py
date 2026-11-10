"""Fit an affine multilabel classifier on frozen features with validation-based selection."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import BudgetSpec
from .metrics import evaluate_scores, format_metrics

class ProbeError(RuntimeError):
    pass

@dataclass
class ProbeOutcome:

    classifier: nn.Linear
    state_dict: Dict[str, torch.Tensor]
    val_metrics: Dict[str, object]
    history: List[Dict[str, object]] = field(default_factory=list)
    epochs_run: int = 0
    best_epoch: int = -1
    stopped_early: bool = False
    degraded: bool = False
    in_dim: int = 0
    num_classes: int = 0

def build_classifier(in_dim: int, num_classes: int, device: str) -> nn.Linear:
    return nn.Linear(in_dim, num_classes).to(device)

def build_optimizer(classifier: nn.Module, budget: BudgetSpec) -> torch.optim.Optimizer:
    if budget.optimizer == "adamw":
        return torch.optim.AdamW(classifier.parameters(), lr=budget.lr,
                                 weight_decay=budget.weight_decay)
    if budget.optimizer == "adam":
        return torch.optim.Adam(classifier.parameters(), lr=budget.lr,
                                weight_decay=budget.weight_decay)
    raise ProbeError(f"unknown optimizer {budget.optimizer!r}")

def build_scheduler(optimizer: torch.optim.Optimizer, budget: BudgetSpec):
    if budget.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=budget.epochs, eta_min=budget.eta_min
        )
    if budget.scheduler == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(budget.milestones), gamma=budget.gamma
        )
    raise ProbeError(f"unknown scheduler {budget.scheduler!r}")

@torch.no_grad()
def predict_logits(classifier: nn.Module, features: torch.Tensor,
                   device: str, batch_size: int = 1024) -> torch.Tensor:
    classifier.eval()
    chunks = []
    for start in range(0, features.shape[0], batch_size):
        chunk = features[start:start + batch_size].to(device, non_blocking=True)
        chunks.append(classifier(chunk).float().cpu())
    if not chunks:
        return torch.zeros(0, classifier.out_features)
    return torch.cat(chunks, dim=0)

def evaluate_features(classifier: nn.Module,
                      features: torch.Tensor,
                      labels: torch.Tensor,
                      label_names: Sequence[str],
                      device: str,
                      threshold: float = 0.5) -> Tuple[Dict[str, object], torch.Tensor]:
    logits = predict_logits(classifier, features, device)
    metrics = evaluate_scores(
        labels.numpy(), logits.numpy(), label_names,
        threshold=threshold, score_space="logit",
    )
    probabilities = torch.sigmoid(logits)
    return metrics, probabilities

def train_linear_probe(train_features: torch.Tensor,
                       train_labels: torch.Tensor,
                       val_features: torch.Tensor,
                       val_labels: torch.Tensor,
                       label_names: Sequence[str],
                       budget: BudgetSpec,
                       device: str = "cuda",
                       seed: int = 42,
                       verbose: bool = True) -> ProbeOutcome:
    in_dim = int(train_features.shape[1])
    num_classes = int(train_labels.shape[1])
    if val_labels.shape[1] != num_classes:
        raise ProbeError(
            f"val labels have {val_labels.shape[1]} classes, train has {num_classes}"
        )
    if train_features.shape[0] != train_labels.shape[0]:
        raise ProbeError("train features/labels row count mismatch")
    if train_features.shape[0] == 0:
        raise ProbeError("empty training set")

    torch.manual_seed(seed)
    np.random.seed(seed)

    classifier = build_classifier(in_dim, num_classes, device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(classifier, budget)
    scheduler = build_scheduler(optimizer, budget)
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_features = train_features.contiguous()
    train_labels = train_labels.contiguous()
    n_train = int(train_features.shape[0])
    batch_size = max(1, min(int(budget.batch_size), n_train))

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metrics: Optional[Dict[str, object]] = None
    best_auc = -float("inf")
    best_epoch = -1
    patience = 0
    history: List[Dict[str, object]] = []
    degraded = False
    stopped_early = False

    finite_val_epochs = 0

    for epoch in range(int(budget.epochs)):
        classifier.train()
        permutation = torch.randperm(n_train, generator=generator)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            index = permutation[start:start + batch_size]
            feats = train_features.index_select(0, index).to(device, non_blocking=True)
            targets = train_labels.index_select(0, index).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(feats)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            if budget.schedule_step == "iteration":
                scheduler.step()
            epoch_loss += float(loss.detach()) * feats.shape[0]
        if budget.schedule_step == "epoch":
            scheduler.step()
        epoch_loss /= max(n_train, 1)

        if budget.early_stop_patience is not None or budget.select == "best_val":
            val_metrics, _ = evaluate_features(
                classifier, val_features, val_labels, label_names, device
            )
            raw_auc = val_metrics["macro_auroc"]
            val_auc = float(raw_auc) if np.isfinite(float(raw_auc)) else float("nan")
            if np.isfinite(val_auc):
                finite_val_epochs += 1
        else:
            val_metrics = {}
            val_auc = float("nan")

        improved = False
        if budget.select == "best_val":
            score = 0.0 if not np.isfinite(val_auc) else val_auc
            if score > best_auc + 1e-12:
                best_auc = score
                best_state = copy.deepcopy(
                    {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
                )
                best_metrics = val_metrics
                best_epoch = epoch
                improved = True
                patience = 0
            else:
                patience += 1

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_macro_auroc": val_auc,
            "val_n_classes_scored": val_metrics.get("n_classes_scored") if val_metrics else None,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
            auc_text = "nan" if not np.isfinite(val_auc) else f"{val_auc:.4f}"
            print(f"    epoch {epoch + 1:3d}/{budget.epochs} loss={epoch_loss:.4f} "
                  f"val_auroc={auc_text} lr={optimizer.param_groups[0]['lr']:.2e}"
                  + ("" if improved else f" patience={patience}"))

        if (budget.early_stop_patience is not None
                and patience >= int(budget.early_stop_patience)):
            stopped_early = True
            if verbose:
                print(f"    early stop at epoch {epoch + 1} "
                      f"(best epoch {best_epoch + 1}, val_auroc={best_auc:.4f})")
            break

    if budget.select == "best_val":
        if best_state is None or finite_val_epochs == 0:
            degraded = True
            best_state = {
                k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()
            }
            best_metrics, _ = evaluate_features(
                classifier, val_features, val_labels, label_names, device
            )
            best_epoch = len(history) - 1
            print(
                "[probe] WARNING: val macro AUROC was undefined in every epoch "
                f"({len(history)} epoch(s)) -> using last-epoch weights"
            )
        classifier.load_state_dict(best_state)
    else:
        best_state = {
            k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()
        }
        best_metrics, _ = evaluate_features(
            classifier, val_features, val_labels, label_names, device
        )
        best_epoch = int(budget.epochs) - 1

    epochs_run = len(history)
    if verbose and best_metrics is not None:
        print(f"    best: {format_metrics(best_metrics)} "
              f"(epoch {best_epoch + 1}/{epochs_run})")
    return ProbeOutcome(
        classifier=classifier,
        state_dict=best_state,
        val_metrics=best_metrics or {},
        history=history,
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        stopped_early=stopped_early,
        degraded=degraded,
        in_dim=in_dim,
        num_classes=num_classes,
    )
