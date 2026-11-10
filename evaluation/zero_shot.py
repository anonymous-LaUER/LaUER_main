"""Evaluate cross-modal retrieval using SGL embeddings."""
import torch
import numpy as np
from tqdm import tqdm
from evaluation.utils import evaluation_autocast

def retrieval_eval(sim_matrix, query2gallery=None, gallery2query=None):
    if query2gallery is None and gallery2query is None:

        N = sim_matrix.shape[0]
        diagonal = np.diag(sim_matrix)
        ranks = 1 + np.sum(sim_matrix > diagonal[:, None], axis=1)
    else:

        ranks = []
        for idx in range(sim_matrix.shape[0]):
            sorted_indices = np.argsort(-sim_matrix[idx])
            if query2gallery is not None:

                gt_ranks = [np.where(sorted_indices == gt)[0][0] for gt in query2gallery[idx]]
                ranks.append(min(gt_ranks) + 1)
            elif gallery2query is not None:
                rank = np.where(sorted_indices == gallery2query[idx])[0][0]
                ranks.append(rank + 1)
        ranks = np.array(ranks)
    r1 = 100.0 * (ranks <= 1).sum() / len(ranks)
    r5 = 100.0 * (ranks <= 5).sum() / len(ranks)
    r10 = 100.0 * (ranks <= 10).sum() / len(ranks)

    return {
        'R@1': r1,
        'R@5': r5,
        'R@10': r10,
        'MedianRank': np.median(ranks),
        'MeanRank': ranks.mean(),
    }

def _get_tokens(model, batch, modality, device):
    key_tok = f'{modality}_tokens'
    if key_tok in batch:
        tok = batch[key_tok]
        return tok.to(device) if isinstance(tok, torch.Tensor) else tok
    encoder = getattr(model, f'{modality}_encoder', None)
    if encoder is None:
        raise RuntimeError(f'{modality} encoder not loaded and no cached tokens in batch')
    if modality == 'text':
        return encoder.encode_tokens(batch['text_input_ids'].to(device),
                                     batch['text_attention_mask'].to(device))
    elif modality == 'image':
        return encoder.encode_tokens(batch['ecg_image'].to(device))
    else:
        return encoder.encode_tokens(batch['ecg_signal'].to(device))

def retrieval_results_from_embeddings(ecg_embeds, image_embeds, text_embeds,
                                      logit_scale):
    scale = logit_scale.detach().float().cpu()
    ecg = ecg_embeds.detach().float().cpu()
    image = image_embeds.detach().float().cpu()
    text = text_embeds.detach().float().cpu()
    sim_ecg_text = (scale * ecg @ text.T).numpy()
    sim_image_text = (scale * image @ text.T).numpy()
    sim_ecg_image = (scale * ecg @ image.T).numpy()
    directions = (
        ('ecg2text', sim_ecg_text), ('text2ecg', sim_ecg_text.T),
        ('image2text', sim_image_text), ('text2image', sim_image_text.T),
        ('ecg2image', sim_ecg_image), ('image2ecg', sim_ecg_image.T),
    )
    return {name: retrieval_eval(sim) for name, sim in directions}

@torch.no_grad()
def zero_shot_retrieval_eval(model, dataloader, device,
                             bf16=False, fp16=False):

    if dataloader is None or len(dataloader) == 0:
        return {}
    was_training = model.training
    model.eval()

    all_ecg_embeds = []
    all_image_embeds = []
    all_text_embeds = []

    for batch in tqdm(dataloader, desc="Zero-shot eval"):
        with evaluation_autocast(device, bf16=bf16, fp16=fp16):
            ecg_tokens = _get_tokens(model, batch, 'ecg', device)
            image_tokens = _get_tokens(model, batch, 'image', device)
            text_tokens = _get_tokens(model, batch, 'text', device)
            ecg_embed, image_embed, text_embed = model.project_features(
                ecg_tokens, image_tokens, text_tokens
            )

        all_ecg_embeds.append(ecg_embed.cpu())
        all_image_embeds.append(image_embed.cpu())
        all_text_embeds.append(text_embed.cpu())

    all_ecg_embeds = torch.cat(all_ecg_embeds, dim=0)
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)

    results = retrieval_results_from_embeddings(
        all_ecg_embeds, all_image_embeds, all_text_embeds,
        model.sgl_loss.logit_scale.exp(),
    )

    model.train(was_training)
    return results

def format_eval_results(results):
    lines = ["=" * 60, "Zero-shot Evaluation Results", "=" * 60]

    for name, metrics in results.items():
        if 'R@1' in metrics:
            lines.append(f"\n  {name}:")
            lines.append(f"    R@1: {metrics['R@1']:.2f}%")
            lines.append(f"    R@5: {metrics['R@5']:.2f}%")
            lines.append(f"    R@10: {metrics['R@10']:.2f}%")
            lines.append(f"    Median Rank: {metrics['MedianRank']:.1f}")
        elif 'ecg_accuracy' in metrics:
            lines.append(f"\n  Classification:")
            lines.append(f"    ECG Acc: {metrics['ecg_accuracy']:.2f}%")
            lines.append(f"    Image Acc: {metrics['image_accuracy']:.2f}%")

    lines.append("=" * 60)
    return "\n".join(lines)
