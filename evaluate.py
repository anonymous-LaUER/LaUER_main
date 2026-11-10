import os
import sys
import argparse
import logging

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import LaUERConfig, load_config_from_yaml
from models.lauer import LaUER
from data.meeti_dataset import create_meeti_dataloaders
from evaluation.zero_shot import zero_shot_retrieval_eval, format_eval_results
from evaluation.cgl_eval import cgl_accuracy_eval
from evaluation.utils import (
    average_eval_results,
    format_full_eval_results,
    load_checkpoint_weights,
)

@torch.no_grad()
def evaluate_sgl(model, dataloader, device, logger, config=None):
    model.eval()
    results = zero_shot_retrieval_eval(
        model, dataloader, device,
        bf16=bool(config and config.train.bf16),
        fp16=bool(config and config.train.fp16),
    )
    logger.info(f"\n{format_eval_results(results)}")
    return results

@torch.no_grad()
def evaluate_full(model, dataloader, device, logger, config=None):
    model.eval()
    bf16 = bool(config and config.train.bf16)
    fp16 = bool(config and config.train.fp16)
    results = zero_shot_retrieval_eval(
        model, dataloader, device, bf16=bf16, fp16=fp16,
    )
    logger.info(f"\n[Zero-Shot]\n{format_eval_results(results)}")

    cgl_results = cgl_accuracy_eval(
        model, dataloader, device, bf16=bf16, fp16=fp16,
    )
    logger.info(
        f"[CGL] Fusion Match Acc={cgl_results['acc_matching']:.4f}"
    )
    return {'zero_shot': results, 'cgl': cgl_results}

def main():
    parser = argparse.ArgumentParser(description='lauer Standalone Evaluation')
    parser.add_argument('--mode', type=str, default='sgl',
                        choices=['sgl', 'sgl+cgl'],
                        help='Model mode: sgl or sgl+cgl')
    parser.add_argument('--config', type=str,
                        default='configs/pretrain_SGL+CGL.yaml',
                        help='Config YAML path')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint (.pt)')
    parser.add_argument('--data_root', type=str, default='./datasets/MEETI',
                        help='MEETI dataset root (overrides config)')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Eval batch size (overrides config; 0=use YAML eval_batch_size)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--test_only', action='store_true',
                        help='Only evaluate on test set (skip val)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    if os.path.exists(args.config):
        config = load_config_from_yaml(args.config)
    else:
        config = LaUERConfig()

    if args.data_root:
        config.data.data_root = args.data_root
    if args.batch_size > 0:
        config.data.eval_batch_size = args.batch_size

        config.data.val_loader.batch_size = args.batch_size

    is_full = (args.mode == 'sgl+cgl')

    device = torch.device(args.device)

    logger.info("=" * 60)
    logger.info(f"lauer Evaluation — Mode: {args.mode}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {device}  |  Data: {config.data.data_root}")
    logger.info("=" * 60)

    from transformers import AutoTokenizer
    tokenizer_path = config.text_encoder.tokenizer_name_or_path
    if not tokenizer_path:
        logger.error("text_encoder.tokenizer_name_or_path not set!")
        return
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    dataloaders = create_meeti_dataloaders(config.data, config.train, tokenizer,
                                          max_text_length=config.text_encoder.max_length,
                                          image_size=config.image_encoder.image_size,
                                          encoder_type=config.image_encoder.encoder_type,
                                          fail_on_error=False)

    val_loader = dataloaders.get('val')
    test_loader = dataloaders.get('test')

    model = LaUER(config, mode=args.mode)

    model.load_pretrained_encoders(device=str(device))
    model = model.to(device)
    model.eval()

    load_checkpoint_weights(model, args.checkpoint, device, logger)

    logger.info(f"\n{'='*40} Evaluation {'='*40}")

    eval_results = {}
    loaders_to_eval = [('Test', test_loader)]
    if not args.test_only:
        loaders_to_eval.insert(0, ('Val', val_loader))

    for loader_name, loader in loaders_to_eval:
        if loader is None:
            logger.info(f"\n--- {loader_name} Set: SKIPPED (no data) ---")
            continue
        logger.info(f"\n--- {loader_name} Set ---")
        if is_full:
            eval_results[loader_name] = evaluate_full(
                model, loader, device, logger, config=config,
            )
        else:
            eval_results[loader_name] = evaluate_sgl(
                model, loader, device, logger, config=config,
            )

    if len(eval_results) == 2:
        logger.info(f"\n{'='*40} Val + Test Averaged {'='*40}")
        avg_result = average_eval_results(eval_results['Val'], eval_results['Test'])
        if isinstance(avg_result, dict) and 'ecg2text' in avg_result:
            logger.info(format_eval_results(avg_result))
        if isinstance(avg_result, dict) and 'cgl' in avg_result:
            logger.info(f"\n[Averaged CGL]\n{format_full_eval_results(avg_result)}")

    logger.info("\nEvaluation complete!")

if __name__ == '__main__':
    main()
