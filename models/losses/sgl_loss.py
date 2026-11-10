"""Semantic-granularity alignment with three bidirectional InfoNCE pairs.

The pair weights follow configuration order (signal-text, image-text,
signal-image); the paper uses (0.4, 0.4, 0.2). Each pair averages both directions."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SGLLoss(nn.Module):
    """Weighted bidirectional alignment of normalized signal, image, and text embeddings."""

    def __init__(self, config):
        super().__init__()
        self.loss_weights = config.loss_weights  # (w_et, w_it, w_ei)

        if config.learnable_temp:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * torch.log(torch.tensor(1.0 / config.temperature))
            )
        else:
            self.register_buffer(
                'logit_scale',
                torch.ones([]) * torch.log(torch.tensor(1.0 / config.temperature))
            )

        self.temp_max = config.temp_max

    def _clamp_temp(self):
        with torch.no_grad():
            self.logit_scale.clamp_(0, math.log(self.temp_max))

    def compute_similarity(self, ecg_embed, image_embed, text_embed):
        with torch.no_grad():
            self._clamp_temp()
            logit_scale = self.logit_scale.exp()
            sim_et = logit_scale * ecg_embed @ text_embed.T
            sim_ei = logit_scale * ecg_embed @ image_embed.T
            sim_it = logit_scale * image_embed @ text_embed.T
        return sim_et, sim_ei, sim_it

    def forward(self, ecg_embed, image_embed, text_embed):
        self._clamp_temp()
        logit_scale = self.logit_scale.exp()

        w_et, w_it, w_ei = self.loss_weights

        sim_et = logit_scale * ecg_embed @ text_embed.T
        sim_ei = logit_scale * ecg_embed @ image_embed.T
        sim_it = logit_scale * image_embed @ text_embed.T

        B = ecg_embed.shape[0]
        labels = torch.arange(B, device=ecg_embed.device, dtype=torch.long)

        loss_ecg_text = (F.cross_entropy(sim_et, labels) + F.cross_entropy(sim_et.T, labels)) / 2
        loss_image_text = (F.cross_entropy(sim_it, labels) + F.cross_entropy(sim_it.T, labels)) / 2
        loss_ecg_image = (F.cross_entropy(sim_ei, labels) + F.cross_entropy(sim_ei.T, labels)) / 2

        total_loss = (w_et * loss_ecg_text +
                      w_it * loss_image_text +
                      w_ei * loss_ecg_image)

        loss_dict = {
            'loss_sgl': total_loss.detach(),
            'loss_ecg_text': loss_ecg_text.detach(),
            'loss_image_text': loss_image_text.detach(),
            'loss_ecg_image': loss_ecg_image.detach(),
            'temperature': (1.0 / logit_scale).detach(),
        }

        return total_loss, loss_dict, (sim_et.detach(), sim_ei.detach(), sim_it.detach())

    def _masked_ce(self, sim, pos_mask):
        """Average log probabilities over all positives for each anchor."""
        log_p = sim - torch.logsumexp(sim, dim=1, keepdim=True)   # log-softmax [Bx, By]
        soft = pos_mask / pos_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return -(soft * log_p).sum(dim=1).mean()

    def forward_masked(self, ecg_embed, image_embed, text_embed,
                       ecg_aug=None, img_aug=None):
        """Compute alignment with explicit positive masks; diagonal masks recover ordinary InfoNCE."""
        self._clamp_temp()
        logit_scale = self.logit_scale.exp()
        B = ecg_embed.shape[0]
        dev = ecg_embed.device

        ecg_anch = torch.cat([ecg_embed, ecg_aug], dim=0) if ecg_aug is not None else ecg_embed
        img_anch = torch.cat([image_embed, img_aug], dim=0) if img_aug is not None else image_embed
        n_ecg, n_img = ecg_anch.shape[0], img_anch.shape[0]

        pid = torch.arange(B, device=dev)
        pid_ecg = torch.cat([pid, pid]) if n_ecg > B else pid
        pid_img = torch.cat([pid, pid]) if n_img > B else pid

        mask_et = (pid_ecg[:, None] == pid[None, :]).float()        # [n_ecg, B]
        mask_it = (pid_img[:, None] == pid[None, :]).float()        # [n_img, B]
        mask_ei = (pid_ecg[:, None] == pid_img[None, :]).float()    # [n_ecg, n_img]

        sim_et = logit_scale * ecg_anch @ text_embed.T
        sim_it = logit_scale * img_anch @ text_embed.T
        sim_ei = logit_scale * ecg_anch @ img_anch.T

        loss_ecg_text = (self._masked_ce(sim_et, mask_et) + self._masked_ce(sim_et.T, mask_et.T)) / 2
        loss_image_text = (self._masked_ce(sim_it, mask_it) + self._masked_ce(sim_it.T, mask_it.T)) / 2
        loss_ecg_image = (self._masked_ce(sim_ei, mask_ei) + self._masked_ce(sim_ei.T, mask_ei.T)) / 2

        w_et, w_it, w_ei = self.loss_weights
        total_loss = (w_et * loss_ecg_text + w_it * loss_image_text
                      + w_ei * loss_ecg_image)

        loss_dict = {
            'loss_sgl': total_loss.detach(),
            'loss_ecg_text': loss_ecg_text.detach(),
            'loss_image_text': loss_image_text.detach(),
            'loss_ecg_image': loss_ecg_image.detach(),
            'temperature': (1.0 / logit_scale).detach(),
        }
        return total_loss, loss_dict
