"""Training configuration summaries and structured run logging."""

import io
import math
import platform
import sys
from contextlib import contextmanager, redirect_stdout

import torch

def _yn(value):
    return "yes" if bool(value) else "no"

def _scalar(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)

def _source(path):
    return path if path else "from scratch"

def _mlp_summary(cfg):
    ln = (
        f"LN(eps={cfg.layernorm:g}, pos={cfg.layernorm_position})"
        if cfg.layernorm > 0 else "LN=off"
    )
    return (
        f"{cfg.input_dim}->{cfg.output_dim}, layers={cfg.num_layers}, "
        f"hidden={cfg.hidden_dim or 'auto'}, act={cfg.activation}, {ln}, "
        f"dropout={cfg.dropout:g}"
    )

def _cgl_weights(config):

    return tuple(config.cgl.loss_weights)

def _selection_formula(config, mode):
    train = config.train
    if mode == "sgl":
        return "average_R@1 / 100"
    terms = [
        f"{train.composite_zeroshot_weight:g}*R@1",
        f"{train.composite_cgl_weight:g}*CGL_Acc",
    ]

    if getattr(train, 'composite_mlc_weight', 0.0) > 0:
        terms.append(f"{train.composite_mlc_weight:g}*mAP")
    if mode == "sgl+cgl+tgl":
        terms.extend([
            f"{train.composite_mlm_weight:g}*MLM_Acc",
            f"{train.composite_ar_weight:g}*AR_Score",
        ])
    return " + ".join(terms)

def _wd_switch_summary(config, mode):
    switches = config.train.wd_switches
    active = [
        ("proj1_sgl", switches.proj1_sgl),
        ("proj2", switches.proj2),
    ]
    if mode != "sgl":
        active.extend([
            ("proj1_cgl", switches.proj1_cgl),
            ("fusion", switches.fusion_encoder),
            ("cgl_head", switches.cgl_head),
        ])
    if not config.image_encoder.freeze:
        active.append(("image_encoder", switches.image_encoder))
    if not config.ecg_encoder.freeze:
        active.append(("ecg_encoder", switches.signal_encoder))
    if mode == "sgl+cgl+tgl":
        active.append(("tgl", switches.tgl))
    return ", ".join(f"{name}={'WD' if enabled else 'no-WD'}" for name, enabled in active)

def objective_formula(config, mode):
    w_et, w_it, w_ei = config.sgl.loss_weights
    sgl = f"L_SGL={w_et:g}*L_ET+{w_it:g}*L_IT+{w_ei:g}*L_EI"
    if mode == "sgl":
        return sgl

    stage_terms = (
        f"{_cgl_weights(config)[0]:g}*L_match+"
        f"{_cgl_weights(config)[1]:g}*L_clf"
    )
    total = (
        f"L_total={config.sgl.sgl_in_global:g}*L_SGL+"
        f"{config.cgl.cgl_in_global:g}*({stage_terms})"
    )
    if mode == "sgl+cgl+tgl":
        lm = (
            f"L_TGL={config.tgl.mlm_loss_weight:g}*L_MLM+"
            f"{config.tgl.ar_loss_weight:g}*L_AR"
        )
        total += f"+{config.tgl.lm_weight:g}*L_TGL"
        return f"{sgl}; {lm}; {total}"
    return f"{sgl}; {total}"

def build_progress_postfix(total_loss, loss_dict, mode, lr):
    """Read displayed GPU losses in one transfer at each progress refresh."""
    wanted = ["total", "loss_sgl"]
    if mode != "sgl":
        wanted += ["loss_matching", "loss_mlc"]
    if mode == "sgl+cgl+tgl":
        wanted += ["loss_mlm", "loss_ar", "mlm_total"]

    cuda_keys, cuda_items, plain = [], [], {}
    for key in wanted:
        value = total_loss if key == "total" else loss_dict.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            if value.is_cuda:
                cuda_keys.append(key)
                cuda_items.append(value.detach().to(torch.float32).reshape(-1))
            else:
                plain[key] = float(value)
        else:
            plain[key] = float(value)
    if cuda_items:
        # one D2H: flatten [N,1] stack to a flat Python list
        readback = torch.stack(cuda_items).reshape(-1).tolist()
        plain.update(dict(zip(cuda_keys, readback)))

    postfix = {
        "total": f"{plain.get('total', 0.0):.3f}",
        "SGL": f"{plain.get('loss_sgl', 0.0):.3f}",
    }
    if mode != "sgl":
        if "loss_matching" in plain:
            postfix["Match"] = f"{plain['loss_matching']:.3f}"
        if "loss_mlc" in plain:
            postfix["MLC"] = f"{plain['loss_mlc']:.3f}"
    if mode == "sgl+cgl+tgl":

        # Display MLM statistics only when masked tokens were evaluated.
        if "loss_mlm" in plain and plain.get("mlm_total", 0.0) > 0:
            postfix["MLM"] = f"{plain['loss_mlm']:.3f}"
        if "loss_ar" in plain:
            postfix["AR"] = f"{plain['loss_ar']:.3f}"
    postfix["lr"] = f"{float(lr):.2e}"
    return postfix

def _encoder_lines(config):
    ecg = config.ecg_encoder
    v = ecg.vit1d
    ecg_detail = (
        f"ViT-1D/{v.model_variant}, width={v.width}, depth={v.depth}, heads={v.heads}, "
        f"input=[{v.lead_num},{v.seq_length}], patch={v.patch_size}, "
        f"pos={v.pos_embedding_type}, dropout={v.dropout:g}, drop_path={v.drop_path_rate:g}, "
        f"weights={_source(v.checkpoint_path)}"
    )

    image = config.image_encoder
    v2 = image.vit2d
    image_detail = (
        f"ViT-2D/{v2.model_variant}, width={v2.width}, depth={v2.depth}, heads={v2.heads}, "
        f"image={v2.image_height}x{v2.image_width}, patch={v2.patch_height}x{v2.patch_width}, "
        f"tokens={v2.num_patches_total + 1}, pos={v2.pos_embedding_type}, "
        f"dropout={v2.dropout:g}, drop_path={v2.drop_path_rate:g}, "
        f"weights={_source(v2.checkpoint_path)}"
    )

    text = config.text_encoder
    return [
        f"  ECG   | {ecg_detail} | trainable={_yn(not ecg.freeze)}",
        f"  Image | {image_detail} | trainable={_yn(not image.freeze)}",
        (
            f"  Text  | MedCPT, hidden={text.hidden_size}, max_length={text.max_length}, "
            f"weights={_source(text.model_name_or_path)} | trainable={_yn(not text.freeze)}"
        ),
    ]

def _projection_lines(config, mode):
    lines = [f"  shared_proj1={_yn(config.projection.use_shared_proj1)}"]
    for name in ("ecg", "image", "text"):
        adapter = getattr(config.projection, name)
        line = (
            f"  {name.upper():5s} SGL | proj1[{_mlp_summary(adapter.sgl.proj1)}] "
            f"-> pool={adapter.sgl.pooling} -> proj2[{_mlp_summary(adapter.sgl.proj2)}] -> L2"
        )
        lines.append(line)
        if mode != "sgl":
            lines.append(
                f"  {name.upper():5s} CGL | proj1[{_mlp_summary(adapter.cgl.proj1)}] -> fusion tokens"
            )
    if mode == "sgl+cgl+tgl":
        d = config.projection.tgl_proj
        lines.append(
            f"  TGL adapter | enabled={_yn(d.enable)}, {_mlp_summary(d)}, "
            f"gradient_to_fused_encoder={_yn(d.allow_tgl_grad_to_encoder)}"
        )
    return lines

def log_run_configuration(logger, config, mode, config_path, data_root, run_dir,
                          device, init_from="",
                          resume="", auto_resume=True, world_size=1):
    train = config.train
    data = config.data
    physical_global_batch = train.batch_size * world_size
    effective_batch = physical_global_batch * train.accumulation_steps
    init_policy = (
        f"stage initialization from {init_from}" if init_from
        else f"resume checkpoint {resume}" if resume
        else "auto-resume checkpoint.pt when present" if auto_resume
        else "new run (auto-resume disabled)"
    )

    _t_nw, _t_pm, _t_pf, _t_persist, _, _t_dl = data.get_train_loader_cfg()
    _v_bs, _v_nw, _v_pm, _v_persist, _, _v_dl = data.get_val_loader_cfg()

    lines = [
        "",
        "=" * 100,
        "LAUER ECG TRAINING CONFIGURATION",
        "=" * 100,
        "[1/8] RUN AND TRAINING OBJECTIVE",
        f"  command={' '.join(sys.argv)}",
        f"  config={config_path}",
        f"  mode={mode} (expected={train.expected_mode or 'not constrained'})",
        f"  goal={_mode_goal(mode)}",
        f"  objective={objective_formula(config, mode)}",
        f"  initialization={init_policy}",
        f"  device={device}, world_size={world_size}, output={run_dir}",
        "[2/8] DATA AND BATCHING",
        f"  data_root={data_root}  (raw mode, no feature cache)",
        (
            f"  train_batch={train.batch_size}/rank, physical_global_batch={physical_global_batch}, "
            f"accumulation={train.accumulation_steps}, effective_global_batch={effective_batch}"
        ),
        (
            f"  SGL_negative_pool={physical_global_batch - 1}/anchor "
            f"(cross-rank gather included; accumulation does not add negatives)"
        ),
        (
            f"  [train_loader] workers={_t_nw}, pin_memory={_t_pm}, "
            f"prefetch_factor={_t_pf}, persistent={_t_persist}, drop_last={_t_dl} | "
            f"[val_loader] batch={_v_bs}, workers={_v_nw}, pin_memory={_v_pm}, "
            f"persistent={_v_persist}, drop_last={_v_dl} | "
            f"text_separator={data.text_separator!r}"
        ),
        "[3/8] ACTIVE ENCODERS",
        *_encoder_lines(config),
        "[4/8] ACTIVE PROJECTIONS AND ADAPTERS",
        *_projection_lines(config, mode),
        "[5/8] SGL / CGL / TGL",
        *_objective_module_lines(config, mode),
        "[6/8] OPTIMIZER AND LEARNING RATES",
        (
            f"  AdamW | base_lr={train.lr:.3e}, min_lr={train.min_lr:.3e}, "
            f"betas={tuple(train.adam_betas)}, eps={train.adam_eps:.1e}"
        ),
        (
            f"  lr_scales | SGL={train.lr_scale_sgl:g} ({train.lr * train.lr_scale_sgl:.3e}), "
            f"CGL={train.lr_scale_cgl:g} ({train.lr * train.lr_scale_cgl:.3e}), "
            f"TGL={train.lr_scale_tgl:g} ({train.lr * train.lr_scale_tgl:.3e}), "
            f"Encoder={train.lr_scale_encoder:g} ({train.lr * train.lr_scale_encoder:.3e})"
        ),
        (
            f"  scheduler={train.scheduler}, epochs={train.epochs}, "
            f"warmup_steps={train.warmup_steps}, warmup_ratio={train.warmup_ratio:g}"
        ),
        (
            f"  weight_decay={train.weight_decay:g}, norm_weight_decay="
            f"{train.norm_weight_decay if train.norm_weight_decay is not None else train.weight_decay:g}"
        ),
        f"  active_wd_policy | {_wd_switch_summary(config, mode)}",
        "[7/8] PRECISION, STABILITY AND REPRODUCIBILITY",
        (
            f"  precision={'BF16' if train.bf16 else 'FP16' if train.fp16 else 'FP32'}, "
            f"grad_clip={train.grad_clip_norm:g}, nan_debug={train.nan_debug}, "
            f"numerics_log_interval={train.nan_log_interval} optimizer steps"
        ),
        (
            f"  AMP | init_scale={train.amp_init_scale:g}, growth_interval={train.amp_growth_interval}, "
            f"backoff={train.amp_backoff_factor:g}"
        ),
        (
            f"  seed={train.seed}, deterministic={train.deterministic}, "
            f"cudnn_benchmark={bool(train.benchmark and not train.deterministic)}"
        ),
        "[8/8] VALIDATION, SELECTION AND ARTIFACTS",
        (
            f"  validate_every={train.val_every_n_epochs} epoch(s), "
            f"early_stop_patience={train.early_stop_patience}"
        ),
        f"  selection_score={_selection_formula(config, mode)}",
        "  best_weights condition=selection score improves AND validation loss decreases",
        f"  tgl_zero_cross_attention_ablation_every={train.tgl_ablation_every_n} epoch(s)",
        "  train.log=configuration + runtime initialization + epoch metrics + warnings/checkpoints",
        "  checkpoint.pt=full resume state (model/optimizer/scheduler/AMP/RNG/progress)",
        "  best_weights.pt/last_weights.pt=stage artifact (weights/mode/components/config/metrics)",
        "=" * 100,
    ]
    logger.info("\n".join(lines))

def log_runtime_environment(logger, device):
    """Log the software/device reality that determines whether YAML precision can run."""
    import torch

    lines = [
        "",
        "[RUNTIME] SOFTWARE AND DEVICE",
        f"  Python={platform.python_version()}, PyTorch={torch.__version__}",
        (
            f"  CUDA runtime={torch.version.cuda or 'unavailable'}, "
            f"cuDNN={torch.backends.cudnn.version() or 'unavailable'}"
        ),
    ]
    if getattr(device, "type", str(device).split(":", 1)[0]) == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        lines.append(
            f"  GPU={props.name}, capability={props.major}.{props.minor}, "
            f"memory={props.total_memory / 1024 ** 3:.2f} GiB, "
            f"BF16_supported={bf16_supported}"
        )
    else:
        lines.append(f"  device={device}, CUDA_available={torch.cuda.is_available()}")
    logger.info("\n".join(lines))

def _mode_goal(mode):
    if mode == "sgl":
        return "align ECG signal, ECG image and report embeddings with tri-pair contrastive learning"
    if mode == "sgl+cgl":
        return "retain SGL alignment and learn signal-image fusion through matching and disease-query supervision"
    return (
        "retain SGL/CGL objectives and add token-granularity report "
        "and explanation generation from the shared fused ECG memory"
    )

def _objective_module_lines(config, mode):
    sgl = config.sgl
    lines = [
        (
            f"  SGL | temperature={sgl.temperature:g}, learnable={_yn(sgl.learnable_temp)}, "
            f"max_logit_scale={sgl.temp_max:g}, pair_weights={list(sgl.loss_weights)}, "
            f"global_weight={sgl.sgl_in_global:g}"
        )
    ]
    if mode == "sgl":
        return lines

    cgl = config.cgl
    s1 = cgl.fusion
    if s1.num_layers > 0:
        fusion = (
            f"alternating {s1.num_layers}L, primary={s1.layer1_primary}, "
            f"cross_kv={list(s1.cross_kv_sequence)}, detach={list(s1.detach_cross_kv)}"
        )
    else:
        fusion = f"simple {s1.simple_num_layers}L"
    ml = cgl.multilabel
    lines.extend([
        (
            f"  CGL | hidden={cgl.hidden_size}, heads={cgl.num_heads}, "
            f"ffn={cgl.intermediate_size}, fusion_dropout={cgl.dropout:g}, "
            f"head_dropout={cgl.cgl_head_dropout:g}, global_weight={cgl.cgl_in_global:g}"
        ),
        (
            f"  CGL fusion | {fusion} (token-preserving)"
        ),
        (
            f"  CGL Query-guided tracing head | query cross-attention {ml.num_layers}L, "
            f"heads={ml.num_heads}, ffn={ml.intermediate_size}, "
            f"query_proj={ml.query_proj_hidden}, loss_weights={list(_cgl_weights(config))} "
            f"[matching, multilabel BCE]"
        ),
    ])

    if mode == "sgl+cgl+tgl":
        d = config.tgl
        lines.extend([
            (
                f"  TGL     | layers={d.num_layers}, hidden={d.hidden_size}, heads={d.num_heads}, "
                f"ffn={d.ffn_size}, vocab={d.vocab_size}, dropout={d.dropout:g}"
            ),
            (
                f"  TGL masking | report_mask={d.report_mask_rate:g}, "
                f"full_mask_schedule={'0.2->0.5 linear' if d.full_mask_rampup else f'{d.full_mask_prob:g} fixed'}"
            ),
            (
                f"  TGL loss | MLM={d.mlm_loss_weight:g}, AR={d.ar_loss_weight:g}, "
                f"global_LM={d.lm_weight:g}, embedding_source={d.medcpt_path}"
            ),
            (
                f"  TGL inference | max_len={d.max_generation_len}, temperature={d.temperature:g}, "
                f"keyword_threshold={d.keyword_threshold:g}"
            ),
        ])
    return lines

def log_dataloader_summary(logger, dataloaders, config, world_size=1):
    train_loader = dataloaders.get("train")
    lines = ["", "[RUNTIME] DATASET AND STEP COUNTS"]
    for split in ("train", "val", "test"):
        loader = dataloaders.get(split)
        if loader is None:
            lines.append(f"  {split:5s} | unavailable")
        else:
            lines.append(
                f"  {split:5s} | samples={len(loader.dataset):,}, batches/rank={len(loader):,}, "
                f"batch_size={loader.batch_size}"
            )
    if train_loader is not None:
        accum = max(int(config.train.accumulation_steps), 1)
        steps_per_epoch = math.ceil(len(train_loader) / accum)
        total_steps = steps_per_epoch * config.train.epochs
        warmup = (
            config.train.warmup_steps if config.train.warmup_steps > 0
            else int(config.train.warmup_ratio * total_steps)
        )
        lines.append(
            f"  optimizer_steps/epoch={steps_per_epoch:,}, total_optimizer_steps={total_steps:,}, "
            f"warmup_steps={warmup:,} ({warmup / max(total_steps, 1) * 100:.2f}%)"
        )
        lines.append(
            f"  effective_global_batch={config.train.batch_size * accum * world_size:,}"
        )
    logger.info("\n".join(lines))

class _LoggerStdout(io.TextIOBase):
    def __init__(self, logger, prefix="  "):
        self.logger = logger
        self.prefix = prefix
        self.buffer = ""

    def write(self, text):
        self.buffer += str(text)
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                self.logger.info(f"{self.prefix}{line.rstrip()}")
        return len(text)

    def flush(self):
        if self.buffer.strip():
            self.logger.info(f"{self.prefix}{self.buffer.rstrip()}")
        self.buffer = ""

@contextmanager
def capture_startup_prints(logger, title):
    """Route startup ``print`` calls to both console and train.log via logger."""
    logger.info(f"\n[RUNTIME] {title}")
    writer = _LoggerStdout(logger)
    with redirect_stdout(writer):
        yield
    writer.flush()

def format_train_epoch(epoch, total_epochs, train_loss, details, config, mode,
                       lr=None, elapsed_seconds=None):
    header = f"[Epoch {epoch:03d}/{total_epochs:03d}] TRAIN | total={train_loss:.4f}"
    if lr is not None:
        header += f" | lr={lr:.3e}"
    if elapsed_seconds is not None:
        header += f" | time={elapsed_seconds:.1f}s"

    lines = [header]
    sgl = float(details.get("loss_sgl", 0.0))
    lines.append(
        f"  SGL    | loss={sgl:.4f}, ET={float(details.get('loss_ecg_text', 0)):.4f}, "
        f"IT={float(details.get('loss_image_text', 0)):.4f}, "
        f"EI={float(details.get('loss_ecg_image', 0)):.4f}, "
        f"temperature={float(details.get('temperature', 0)):.5f}"
    )

    cgl_contribution = 0.0
    if mode != "sgl":
        w1, w2 = _cgl_weights(config)
        loss_matching = float(details.get("loss_matching", 0.0))
        acc_matching = float(details.get("acc_matching", 0.0))
        loss_mlc = float(details.get("loss_mlc", 0.0))
        cgl_contribution = w1 * loss_matching + w2 * loss_mlc
        cgl_contribution *= config.cgl.cgl_in_global
        mlc_p = float(details.get("mlc_precision", 0.0))
        mlc_r = float(details.get("mlc_recall", 0.0))
        lines.append(
            f"  CGL    | Match={loss_matching:.4f} (acc={acc_matching * 100:.1f}%), "
            f"concept_BCE={loss_mlc:.4f} (P={mlc_p * 100:.1f}% R={mlc_r * 100:.1f}%), "
            f"weighted_contribution={cgl_contribution:.4f}"
        )

    lm_contribution = 0.0
    if mode == "sgl+cgl+tgl" and "loss_ar" in details:
        ar = float(details.get("loss_ar", 0.0))
        ar_ppl = math.exp(min(ar, 20.0))
        lm_contribution = config.tgl.lm_weight * (
            config.tgl.mlm_loss_weight * float(details.get("loss_mlm", 0.0))
            + config.tgl.ar_loss_weight * ar
        )
        if config.tgl.effective_loss_mode == 'ar':

            rep_t = float(details.get("report_total", 0.0))
            rep_c = float(details.get("report_correct", 0.0))
            expl_t = float(details.get("expl_total", 0.0))
            expl_c = float(details.get("expl_correct", 0.0))
            lines.append(
                f"  TGL     | AR={ar:.4f}, AR_PPL={ar_ppl:.2f}, "
                f"ReportAcc={rep_c / max(rep_t, 1.0) * 100:.1f}%, "
                f"ExplAcc={expl_c / max(expl_t, 1.0) * 100:.1f}%, "
                f"weighted_contribution={lm_contribution:.4f}"
            )
        else:
            mlm = float(details.get("loss_mlm", 0.0))
            mlm_total = float(details.get("mlm_total", 0.0))
            mlm_acc = float(details.get("mlm_correct", 0.0)) / max(mlm_total, 1.0)

            sep_total = float(details.get("sep_total", 0.0))
            sep_acc = float(details.get("sep_correct", 0.0)) / max(sep_total, 1.0)
            sep_str = f"SEP={sep_acc * 100:.1f}%, " if sep_total > 0 else ""
            epoch_progress = (epoch - 1) / max(total_epochs - 1, 1)
            full_mask_prob = (
                min(0.5, 0.2 + 0.3 * epoch_progress)
                if config.tgl.full_mask_rampup else config.tgl.full_mask_prob
            )
            lines.append(
                f"  TGL     | MLM={mlm:.4f} (acc={mlm_acc * 100:.1f}%), "
                f"{sep_str}AR={ar:.4f}, "
                f"AR_PPL={ar_ppl:.2f}, full_mask={full_mask_prob:.3f}, "
                f"weighted_contribution={lm_contribution:.4f}"
            )

    contributions = [f"SGL={config.sgl.sgl_in_global * sgl:.4f}"]
    if mode != "sgl":
        contributions.append(f"CGL={cgl_contribution:.4f}")
    if mode == "sgl+cgl+tgl":
        contributions.append(f"LM={lm_contribution:.4f}")
    lines.append(f"  Objective contributions | {', '.join(contributions)}")
    return "\n".join(lines)

def format_validation_epoch(epoch, total_epochs, val_results, config, mode):
    details = val_results.get("loss_details", {})
    retrieval_component = (
        val_results["avg_r1"] / 100.0
        if mode == "sgl"
        else config.train.composite_zeroshot_weight * val_results["avg_r1"] / 100.0
    )
    lines = [
        f"[Epoch {epoch:03d}/{total_epochs:03d}] VALID | total={val_results['val_loss']:.4f}",
        (
            f"  Retrieval | average_R@1={val_results['avg_r1']:.2f}% | "
            f"selection_component={retrieval_component:.4f}"
        ),
    ]
    if mode != "sgl":
        stages = []
        if "acc_matching" in details:
            stages.append(f"Match={float(details['acc_matching']) * 100:.1f}%")

        lines.append(
            f"  CGL      | {', '.join(stages)}, average={val_results['avg_itm'] * 100:.1f}%"
        )

    if val_results.get("multilabel"):
        ml = val_results["multilabel"]
        lines.append(
            f"  MultiLabel| mAP={ml['mlc_map']:.4f}, macro-F1@0.5={ml['mlc_macro_f1']:.4f}, "
            f"N={ml['mlc_n']}"
        )
    if val_results.get("tgl_metrics"):
        dm = val_results["tgl_metrics"]
        if 'mlm_accuracy' in dm:
            lines.append(
                f"  TGL       | MLM={dm['mlm_loss']:.4f}, MLM_acc={dm['mlm_accuracy'] * 100:.1f}%, "
                f"AR={dm['ar_loss']:.4f}, AR_PPL={dm['ar_ppl']:.2f}"
            )
        else:

            lines.append(
                f"  TGL       | AR={dm['ar_loss']:.4f}, AR_PPL={dm['ar_ppl']:.2f}, "
                f"ReportAcc={dm.get('report_accuracy', 0.0) * 100:.1f}%, "
                f"ExplAcc={dm.get('expl_accuracy', 0.0) * 100:.1f}%"
            )
    lines.append(
        f"  Selection | score={val_results['composite']:.4f}, "
        f"formula={_selection_formula(config, mode)}"
    )
    return "\n".join(lines)
