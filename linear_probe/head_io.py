"""Persist the exact selected linear probe; no separate training implementation."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

def head_identity(encoder_tag, preprocess, cohort, task, mode, feature_kind,
                  ratio, budget, seed):
    return dict(encoder_tag=encoder_tag, preprocess_tag=preprocess.tag(),
                cohort_fingerprint=cohort.fingerprint, label_names=list(cohort.label_names),
                label_order=cohort.label_order, task=task, mode=mode,
                feature_kind="fusion_cls" if mode == "fusion" else feature_kind,
                ratio=float(ratio), budget=budget, seed=int(seed))

def head_path(cache_dir, identity):
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return Path(cache_dir) / "probe_heads" / identity["encoder_tag"] / (
        "%s__%s__r%g__%s__seed%d__%s.pt" % (
            identity["task"], identity["mode"], identity["ratio"],
            identity["budget"], identity["seed"], digest))

def save_head(cache_dir, identity, outcome, test_metrics, selection):
    path = head_path(cache_dir, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format_version=1, identity=identity,
                   state_dict={k: v.detach().cpu().clone() for k, v in outcome.state_dict.items()},
                   in_dim=outcome.in_dim, num_classes=outcome.num_classes,
                   best_epoch=outcome.best_epoch, selection=selection,
                   degraded=outcome.degraded, val_metrics=outcome.val_metrics,
                   test_metrics=test_metrics)
    # Metrics can contain numpy scalars; persist only safe basic scalar types.
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if hasattr(value, "item") and not isinstance(value, torch.Tensor):
            return value.item()
        return value
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(clean(payload), tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path.resolve()

def load_head(path, identity, in_dim=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("identity") != identity:
        raise ValueError("linear head provenance mismatch: %s" % path)
    width = payload["in_dim"]
    classes = len(identity["label_names"])
    state = payload["state_dict"]
    if (in_dim is not None and width != in_dim) or payload["num_classes"] != classes:
        raise ValueError("linear head feature/label dimensions mismatch")
    if set(state) != {"weight", "bias"} or state["weight"].shape != (classes, width) or state["bias"].shape != (classes,):
        raise ValueError("invalid linear head state")
    if not all(torch.isfinite(t).all() for t in state.values()):
        raise ValueError("non-finite linear head")
    model = torch.nn.Linear(width, classes)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model, payload
