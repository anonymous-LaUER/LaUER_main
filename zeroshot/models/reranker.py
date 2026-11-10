
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import logging
from typing import Dict, List, Sequence

import torch

LOGGER = logging.getLogger(__name__)

def task_cgl_key(task_name: str) -> str:
    return f"concept_logit__{task_name}"

@dataclass
class PromptFeatures:
    labels: List[str]
    prompts: List[str]
    sgl_embeddings: torch.Tensor
    query_embeddings: torch.Tensor
    cgl_prompts: List[str] | None = None
    truncation: Dict | None = None

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def select(self, labels: Sequence[str]) -> "PromptFeatures":
        indices = [self.labels.index(label) for label in labels]
        return PromptFeatures(
            list(labels), [self.prompts[i] for i in indices],
            self.sgl_embeddings[indices], self.query_embeddings[indices],
            [self.cgl_prompts[i] for i in indices] if self.cgl_prompts is not None else None,
        )

class LaUERZeroshotScorer:

    def __init__(self, model, use_amp: bool = True) -> None:
        self.model = model
        self.device = model.device
        self.use_amp = bool(use_amp and self.device.type == "cuda")

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=True
        )

    def _ensure_query_encoder(self):
        if getattr(self.model, 'query_encoder', None) is None:
            query_encoder = self.model._rebuild_query_encoder()
            query_encoder.load_model()
            query_encoder.to(self.model.device)
        else:
            query_encoder = self.model.query_encoder
        query_encoder.load_model()
        query_encoder.to(self.device)
        query_encoder.eval()
        return query_encoder

    def _tokenize_prompts(self, encoder, prompts, max_length, branch):
        if max_length < 2:
            raise ValueError(f"{branch} max_length must be at least 2")
        position_limit = encoder._transformer.config.max_position_embeddings
        if max_length > position_limit:
            raise ValueError(f"{branch} max_length={max_length} exceeds {position_limit}")
        tokenizer = encoder.tokenizer
        lengths = [len(ids) for ids in tokenizer(
            list(prompts), add_special_tokens=True, truncation=False,
        )["input_ids"]]
        truncated = [i for i, length in enumerate(lengths) if length > max_length]
        LOGGER.info("%s prompts: %d, maximum tokens=%d, limit=%d, truncated=%d",
                    branch, len(prompts), max(lengths), max_length, len(truncated))
        if truncated:
            LOGGER.warning("%s truncated prompt indices: %s", branch, truncated)
        encoded = tokenizer(
            list(prompts), add_special_tokens=True, padding="longest",
            truncation=True, max_length=max_length, return_tensors="pt",
        )
        return encoded, {"max_length": max_length, "token_lengths": lengths,
                         "truncated_indices": truncated}

    @torch.inference_mode()
    def encode_prompts(
        self,
        labels: Sequence[str],
        prompts: Sequence[str],
        max_length: int | None = None,
        *,
        cgl_prompts: Sequence[str] | None = None,
        sgl_max_length: int = 512,
        cgl_max_length: int = 512,
    ) -> PromptFeatures:
        if len(labels) != len(prompts):
            raise ValueError("labels and prompts must have the same length")
        if not labels:
            raise ValueError("At least one class prompt is required")
        if len(set(labels)) != len(labels):
            raise ValueError("Prompt labels must be unique")
        cgl_prompts = list(prompts if cgl_prompts is None else cgl_prompts)
        if len(cgl_prompts) != len(labels):
            raise ValueError("labels and cgl_prompts must have the same length")
        if max_length is not None:  # legacy explicit limit applies to both branches
            sgl_max_length = cgl_max_length = max_length
        article_encoder = self.model.text_encoder
        article_encoder.load_model()
        article_encoder.to(self.device)
        article_encoder.eval()
        article_ids, article_audit = self._tokenize_prompts(
            article_encoder, prompts, sgl_max_length, "SGL/Article")
        article_mask = article_ids["attention_mask"].to(self.device)
        with self._autocast():
            article_tokens = article_encoder.encode_tokens(
                article_ids["input_ids"].to(self.device), article_mask)
            sgl_embeddings = self.model.adapters.text_adapter.forward_sgl(
                article_tokens, article_mask.bool())
        # Query CLS is reserved for the multi-label head, as during training.
        query_encoder = self._ensure_query_encoder()
        encoded, query_audit = self._tokenize_prompts(
            query_encoder, cgl_prompts, cgl_max_length, "CGL/Query")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with self._autocast():
            query_cls = query_encoder(input_ids, attention_mask)  # [C, 768]

        self.model.query_encoder = None
        self.model._query_encoder_released = True
        return PromptFeatures(
            labels=list(labels),
            prompts=list(prompts),
            sgl_embeddings=sgl_embeddings,
            query_embeddings=query_cls,
            cgl_prompts=cgl_prompts,
            truncation={"sgl": article_audit, "cgl": query_audit},
        )

    @torch.inference_mode()
    def score_task_sets(self, encoded, prompts, task_labels):
        """Reuse ECG features, but never mix queries from different tasks."""
        result = {key: value.detach().cpu()
                  for key, value in self._coarse_scores(encoded, prompts).items()}
        result["candidate_order"] = torch.argsort(
            result["signal_image_sgl"], dim=1, descending=True)
        fusion = encoded["fusion_logits"].float()
        result.update({
            "fusion_match_logit": fusion[:, 1].cpu(),
            "fusion_margin": (fusion[:, 1] - fusion[:, 0]).cpu(),
            "fusion_probability": torch.softmax(fusion, dim=-1)[:, 1].cpu(),
        })
        for task_name, labels in task_labels.items():
            features = prompts.select(labels)
            with self._autocast():
                logits = self.model.query_guided_tracing_head(
                    features.query_embeddings.float(), encoded["fused_si"].float())
            result[task_cgl_key(task_name)] = logits.float().cpu()
        return result

    @torch.inference_mode()
    def encode_signal_image(
        self,
        ecg_signal: torch.Tensor,
        ecg_image: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ecg_signal = ecg_signal.to(self.device, non_blocking=True)
        ecg_image = ecg_image.to(self.device, non_blocking=True)
        with self._autocast():
            ecg_tokens = self.model.encode_ecg(ecg_signal)
            image_tokens = self.model.encode_image(ecg_image)
            ecg_sgl = self.model.adapters.ecg_adapter.forward_sgl(ecg_tokens)
            image_sgl = (
                self.model.adapters.image_adapter.forward_sgl(image_tokens)
            )
            ecg_cgl = self.model.adapters.ecg_adapter.forward_cgl(ecg_tokens)
            image_cgl = (
                self.model.adapters.image_adapter.forward_cgl(image_tokens)
            )
            if self.model._use_alternating_fusion:
                fused_si = self.model._run_alternating_fusion(
                    ecg_cgl, image_cgl
                )
            else:
                fused_si = self.model.fusion_si(
                    primary=ecg_cgl, cross_kv=image_cgl
                )
            fusion_logits = self.model.cgl_loss_module.heads[0](
                fused_si[:, 0, :]
            )
        return {
            "ecg_sgl": ecg_sgl,
            "image_sgl": image_sgl,
            "fused_si": fused_si,
            "fusion_logits": fusion_logits,
        }

    def _coarse_scores(
        self,
        encoded: Dict[str, torch.Tensor],
        prompts: PromptFeatures,
    ) -> Dict[str, torch.Tensor]:
        scale = self.model.sgl_loss.logit_scale.detach().float().exp()
        text = prompts.sgl_embeddings.float()
        ecg = encoded["ecg_sgl"].float()
        image = encoded["image_sgl"].float()
        ecg_text = scale * (ecg @ text.T)
        image_text = scale * (image @ text.T)
        w_ecg, w_image, _ = self.model.config.sgl.loss_weights
        denominator = float(w_ecg + w_image)
        if denominator <= 0:
            raise ValueError("SGL ECG/text + image/text weights must be positive")
        combined = (
            float(w_ecg) * ecg_text + float(w_image) * image_text
        ) / denominator
        return {
            "ecg_sgl": ecg_text,
            "image_sgl": image_text,
            "signal_image_sgl": combined,
        }

    @torch.inference_mode()
    def score_encoded(
        self,
        encoded: Dict[str, torch.Tensor],
        prompts: PromptFeatures,
        top_k: int | None = None,
        candidate_chunk_size: int = 4,
    ) -> Dict[str, torch.Tensor]:
        coarse = self._coarse_scores(encoded, prompts)
        batch_size, num_classes = coarse["signal_image_sgl"].shape

        candidate_order = torch.argsort(
            coarse["signal_image_sgl"], dim=1, descending=True
        )

        with self._autocast():
            concept_logits = self.model.query_guided_tracing_head(
                prompts.query_embeddings.float(),
                encoded["fused_si"].float(),
            ).float()

        fusion_logits = encoded["fusion_logits"].float()
        result = {
            **coarse,
            "concept_logit": concept_logits,
            "candidate_order": candidate_order,
            "candidate_mask": torch.isfinite(concept_logits),
            "fusion_match_logit": fusion_logits[:, 1],
            "fusion_margin": fusion_logits[:, 1] - fusion_logits[:, 0],
            "fusion_probability": torch.softmax(fusion_logits, dim=-1)[:, 1],
            "logit_scale": self.model.sgl_loss.logit_scale.detach().float().exp(),
        }
        return {
            key: value.detach().cpu()
            if isinstance(value, torch.Tensor)
            else value
            for key, value in result.items()
        }

    @torch.inference_mode()
    def score_batch(
        self,
        ecg_signal: torch.Tensor,
        ecg_image: torch.Tensor,
        prompts: PromptFeatures,
        top_k: int | None = None,
        candidate_chunk_size: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """Encode and score a batch; returned tensors live on CPU."""
        encoded = self.encode_signal_image(ecg_signal, ecg_image)
        return self.score_encoded(
            encoded,
            prompts,
            top_k=top_k,
            candidate_chunk_size=candidate_chunk_size,
        )
