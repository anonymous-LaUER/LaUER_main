"""Reproducibility helpers shared by single-process and DDP training."""

import os
import random
from collections.abc import Mapping

import numpy as np
import torch

_SPLIT_OFFSETS = {
    "train": 0,
    "val": 10_000,
    "test": 20_000,
}

def _normalize_seed(seed):
    """Keep derived seeds inside the range accepted by torch.Generator."""
    return int(seed) % (2**63 - 1)

def seed_rng(seed):
    """Seed Python, NumPy and PyTorch without changing backend policy."""
    seed = _normalize_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

def configure_reproducibility(seed, deterministic=False, benchmark=True):
    """Apply the configured backend policy and seed every process-global RNG.

    cuDNN benchmark and deterministic execution are mutually exclusive.  When
    deterministic execution is requested it always wins, regardless of the
    YAML benchmark value.
    """
    deterministic = bool(deterministic)
    effective_benchmark = bool(benchmark) and not deterministic

    if deterministic:
        # Required by deterministic CUDA matrix multiplication on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = effective_benchmark
    torch.use_deterministic_algorithms(deterministic)
    seed = seed_rng(seed)
    return {
        "seed": seed,
        "deterministic": deterministic,
        "benchmark": effective_benchmark,
    }

def data_seed(base_seed, split="train", rank=0):
    """Derive an independent, stable seed for one split and DDP rank."""
    if split not in _SPLIT_OFFSETS:
        raise ValueError(f"Unknown data split: {split!r}")
    return _normalize_seed(int(base_seed) + _SPLIT_OFFSETS[split] + int(rank) * 100_000)

def make_data_generator(base_seed, split="train", rank=0):
    generator = torch.Generator()
    generator.manual_seed(data_seed(base_seed, split=split, rank=rank))
    return generator

def seed_worker(worker_id):
    """Seed third-party RNGs from the worker seed assigned by DataLoader."""
    del worker_id  # The assigned torch seed already includes the worker id.
    worker_seed = torch.initial_seed()
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)

def loader_generators(dataloaders):
    """Collect named DataLoader generators, omitting missing loaders."""
    return {
        name: loader.generator
        for name, loader in dataloaders.items()
        if loader is not None and getattr(loader, "generator", None) is not None
    }

def _capture_generator_states(generators):
    if generators is None:
        return None
    if isinstance(generators, Mapping):
        return {name: generator.get_state() for name, generator in generators.items()}
    return generators.get_state()

def capture_rng_state(generators=None):
    """Capture global RNGs plus optional DataLoader generator state."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        # Store only this process's active CUDA device.  In DDP, querying all
        # devices can initialize RNG contexts (and memory) on GPUs owned by
        # other ranks.
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "data_generators": _capture_generator_states(generators),
    }

def _restore_generator_states(saved, generators):
    if saved is None or generators is None:
        return
    if isinstance(generators, Mapping):
        if not isinstance(saved, Mapping):
            return
        for name, state in saved.items():
            generator = generators.get(name)
            if generator is not None:
                generator.set_state(state)
        return
    if not isinstance(saved, Mapping):
        generators.set_state(saved)

def restore_rng_state(state, generators=None, logger=None):
    """Restore a state produced by :func:`capture_rng_state`.

    Older checkpoints may not contain all fields; those fields are left at the
    freshly seeded value instead of making resume fail.
    """
    if not state:
        return False
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])

    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        if isinstance(cuda_state, torch.Tensor):
            torch.cuda.set_rng_state(cuda_state)
        elif len(cuda_state) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_state)
        elif len(cuda_state) > torch.cuda.current_device():
            # Old DDP checkpoints stored every visible device's state.
            torch.cuda.set_rng_state(cuda_state[torch.cuda.current_device()])
        elif logger is not None:
            logger.warning(
                "[Seed] CUDA RNG state was not restored because checkpoint and "
                "current CUDA device counts differ."
            )

    # Backward compatibility with a short-lived key name used by older runs.
    saved_generators = state.get("data_generators", state.get("data_generator"))
    _restore_generator_states(saved_generators, generators)
    return True
