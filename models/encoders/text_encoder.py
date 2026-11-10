"""Frozen MedCPT backbones for clinical text and disease-query features."""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size          # 768 (BERT-base)
        self.max_length = config.max_length
        self.pooler_type = "cls_pooler"
        self._transformer = None
        self._tokenizer = None
        self._loaded = False

    def load_model(self):
        if self._loaded:
            print(f"[TextEncoder] Already loaded, skipping.")
            return

        model_path = getattr(self.config, 'model_name_or_path', None) or getattr(self.config, 'model_name', None)
        if not model_path:
            raise ValueError("[TextEncoder] model_name_or_path not set in config!")

        tok_path = (getattr(self.config, 'tokenizer_name_or_path', None) or
                    getattr(self.config, 'tokenizer_name', None) or model_path)

        print(f"[TextEncoder] Loading from local path: {model_path}")
        print(f"[TextEncoder] Loading tokenizer from: {tok_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
        self._transformer = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
        )

        if self.config.freeze:
            self._transformer.requires_grad_(False)
            self._transformer.eval()

        self._loaded = True
        print(f"[TextEncoder] Loaded: {model_name_display(model_path)}, "
              f"hidden_size={self._transformer.config.hidden_size}, "
              f"freeze={self.config.freeze}")

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            tok_path = (getattr(self.config, 'tokenizer_name_or_path', None) or
                        getattr(self.config, 'tokenizer_name', None))
            self._tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
        return self._tokenizer

    def _pool(self, transformer_output, attention_mask):
        return transformer_output.last_hidden_state[:, 0, :]

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None):
        pooled, _ = self.encode_both(input_ids, attention_mask)
        return pooled

    @torch.no_grad()
    def encode_tokens(self, input_ids, attention_mask=None):
        _, tokens = self.encode_both(input_ids, attention_mask)
        return tokens

    # Language features are frozen; downstream projection layers remain trainable.
    @torch.no_grad()
    def encode_both(self, input_ids, attention_mask=None):
        if not self._loaded:
            self.load_model()

        if attention_mask is None:
            attention_mask = (input_ids != self._tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        output = self._transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        tokens = output.last_hidden_state       # [B, L, 768]
        pooled = self._pool(output, attention_mask)  # [B, 768]
        return pooled, tokens

    def tokenize(self, text_list, max_length=None, padding='max_length', truncation=True):
        if max_length is None:
            max_length = self.max_length

        encoded = self.tokenizer(
            text_list,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors='pt',
        )
        return encoded

    @property
    def device(self):
        if self._transformer is not None:
            return next(self._transformer.parameters()).device
        return torch.device('cpu')

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        if self._transformer is not None:
            self._transformer = self._transformer.to(*args, **kwargs)
        return result

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze and self._transformer is not None:
            self._transformer.eval()
        return self

def model_name_display(path: str) -> str:
    if '/' in path or '\\' in path:
        import os
        return os.path.basename(path)
    return path
