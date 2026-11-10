"""Evaluate the signal-image matching component of CGL."""
import torch

from tqdm import tqdm
from models.losses.cgl_loss import hard_negative_mining
from evaluation.utils import evaluation_autocast

def _encode_from_batch(model, batch, device):
    ecg_tokens = model.ecg_encoder.encode_tokens(batch['ecg_signal'].to(device))
    image_tokens = model.image_encoder.encode_tokens(batch['ecg_image'].to(device))
    text_tokens = model.text_encoder.encode_tokens(
        batch['text_input_ids'].to(device),
        batch['text_attention_mask'].to(device),
    )
    text_mask = batch['text_attention_mask'].to(device).bool()

    return ecg_tokens, image_tokens, text_tokens, text_mask

def _fusion_accuracy(model, ecg_proj, img_proj, sim_ei):
    neg_img_idx = hard_negative_mining(sim_ei)
    neg_sig_idx = hard_negative_mining(sim_ei.T)

    sig_all = torch.cat([
        ecg_proj, ecg_proj, ecg_proj[neg_sig_idx],
    ], dim=0)
    img_all = torch.cat([
        img_proj, img_proj[neg_img_idx], img_proj,
    ], dim=0)

    cls_si_pos, cls_si_neg1, cls_si_neg2, fused_si_pos = model.forward_cgl_fusion(
        sig_all, img_all
    )
    _, acc1 = model.cgl_loss_module.compute_stage_loss(
        1, cls_si_pos, cls_si_neg1, cls_si_neg2
    )
    return acc1, fused_si_pos

@torch.no_grad()
def cgl_accuracy_eval(model, dataloader, device, bf16=False, fp16=False):
    was_training = model.training
    with evaluation_autocast(device, bf16=bf16, fp16=fp16):
        result = _cgl_accuracy_eval_impl(model, dataloader, device)
    model.train(was_training)
    return result

@torch.no_grad()
def _cgl_accuracy_eval_impl(model, dataloader, device):
    model.eval()

    total_acc1 = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="CGL Eval", unit="batch"):
        ecg_tokens, image_tokens, text_tokens, text_mask = _encode_from_batch(
            model, batch, device
        )
        B = ecg_tokens.size(0)
        if B < 2:
            continue

        (ecg_embed, image_embed, _), \
            (ecg_proj, img_proj, _) = model.adapters.forward_combined(
            ecg_tokens, image_tokens, text_tokens
        )

        logit_scale = model.sgl_loss.logit_scale.exp()
        sim_ei = logit_scale * ecg_embed @ image_embed.T

        acc1, _ = _fusion_accuracy(model, ecg_proj, img_proj, sim_ei)
        total_acc1 += acc1
        num_batches += 1

    avg_acc1 = total_acc1 / max(num_batches, 1)
    if isinstance(avg_acc1, torch.Tensor):
        avg_acc1 = float(avg_acc1.item())

    return {
        'acc_matching': avg_acc1,
        'num_batches': num_batches,
    }

