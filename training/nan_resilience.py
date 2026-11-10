"""Finite-value checks, bounded learning-rate recovery, and resumable failure diagnostics."""

import logging
import math

logger = logging.getLogger(__name__)

class NaNResilienceManager:

    def __init__(self, lr_reduce_factor=0.5, max_consecutive_skip=5,
                 max_consecutive_fatal=15, max_lr_reductions=3, min_lr=1e-8,
                 max_consecutive_amp_overflow=5, max_amp_overflow_per_epoch=20):

        self.consecutive_nan = 0
        self.total_nan_in_epoch = 0
        self.lr_reductions_total = 0
        self.consecutive_amp_overflow = 0
        self.total_amp_overflow_in_epoch = 0

        self.weight_nan_detected = False
        self.grad_nan_detected = False

        self.first_nan_module = None
        self.first_nan_step = -1
        self.last_optimizer_status = None

        self.lr_reduce_factor = lr_reduce_factor
        self.max_consecutive_skip = max_consecutive_skip
        self.max_consecutive_fatal = max_consecutive_fatal
        self.max_lr_reductions = max_lr_reductions
        self.min_lr = min_lr
        self.max_consecutive_amp_overflow = max_consecutive_amp_overflow
        self.max_amp_overflow_per_epoch = max_amp_overflow_per_epoch

    def register_nan_batch(self, loss_dict=None, module_name=None, step=None):
        self.consecutive_nan += 1
        self.total_nan_in_epoch += 1

        if self.first_nan_module is None and module_name is not None:
            self.first_nan_module = module_name
            self.first_nan_step = step

    def register_clean_batch(self):
        self.consecutive_nan = 0

    def register_clean_optimizer_step(self):
        self.consecutive_amp_overflow = 0

    def register_amp_overflow(self, param_names=None, module_name=None):
        self.consecutive_amp_overflow += 1
        self.total_amp_overflow_in_epoch += 1

        if module_name is not None and self.first_nan_module is None:
            self.first_nan_module = module_name

    def register_weight_nan(self, param_names):
        self.weight_nan_detected = True
        if param_names and self.first_nan_module is None:
            self.first_nan_module = param_names[0]

    def register_grad_nan(self, param_names):
        self.grad_nan_detected = True
        if param_names and self.first_nan_module is None:
            self.first_nan_module = param_names[0]

    @property
    def action(self):

        if self.weight_nan_detected or self.grad_nan_detected:
            if self.lr_reductions_total < self.max_lr_reductions:
                return 'rollback'
            return 'fatal_stop'

        if (self.consecutive_amp_overflow >= self.max_consecutive_amp_overflow or
                self.total_amp_overflow_in_epoch >= self.max_amp_overflow_per_epoch):
            if self.lr_reductions_total < self.max_lr_reductions:
                return 'rollback'
            return 'fatal_stop'

        if self.consecutive_nan >= self.max_consecutive_fatal:
            if self.lr_reductions_total < self.max_lr_reductions:
                return 'rollback'
            return 'fatal_stop'

        if self.consecutive_nan >= self.max_consecutive_skip:
            if self.lr_reductions_total < self.max_lr_reductions:
                return 'reduce_lr'
            return 'rollback'

        return 'skip'

    @property
    def is_fatal(self):
        return self.action in ('rollback', 'fatal_stop')

    def reduce_lr(self, optimizer, scheduler=None):
        lr_changes = []
        for i, pg in enumerate(optimizer.param_groups):
            old_lr = pg['lr']
            new_lr = max(old_lr * self.lr_reduce_factor, self.min_lr)
            pg['lr'] = new_lr
            lr_changes.append((old_lr, new_lr))

        if scheduler is not None:
            scheduler.base_lrs = [
                lr * self.lr_reduce_factor for lr in scheduler.base_lrs
            ]
            if hasattr(scheduler, '_last_lr'):
                scheduler._last_lr = [pg['lr'] for pg in optimizer.param_groups]

        self.lr_reductions_total += 1
        self.consecutive_nan = 0
        self.weight_nan_detected = False
        self.grad_nan_detected = False

        return lr_changes

    def register_rollback_lr_reduction(self):
        self.lr_reductions_total += 1

    def reset_epoch(self):
        self.consecutive_nan = 0
        self.total_nan_in_epoch = 0
        self.weight_nan_detected = False
        self.grad_nan_detected = False
        self.consecutive_amp_overflow = 0
        self.total_amp_overflow_in_epoch = 0
        self.first_nan_module = None
        self.first_nan_step = -1
        self.last_optimizer_status = None

    def diagnosis(self):
        parts = [
            f"consecutive={self.consecutive_nan}",
            f"total_in_epoch={self.total_nan_in_epoch}",
            f"lr_reductions={self.lr_reductions_total}",
            f"weight_nan={self.weight_nan_detected}",
            f"grad_nan={self.grad_nan_detected}",
            f"amp_overflow_consecutive={self.consecutive_amp_overflow}",
            f"amp_overflow_total={self.total_amp_overflow_in_epoch}",
        ]
        if self.first_nan_module:
            parts.append(f"first_source={self.first_nan_module}")
        if self.first_nan_step >= 0:
            parts.append(f"first_step={self.first_nan_step}")
        parts.append(f"action={self.action}")
        return " | ".join(parts)

    def diagnostic_state(self):
        return {
            'diagnosis': self.diagnosis(),
            'consecutive_nan': self.consecutive_nan,
            'total_nan_in_epoch': self.total_nan_in_epoch,
            'consecutive_amp_overflow': self.consecutive_amp_overflow,
            'total_amp_overflow_in_epoch': self.total_amp_overflow_in_epoch,
            'lr_reductions_total': self.lr_reductions_total,
            'first_nan_module': self.first_nan_module,
            'first_nan_step': self.first_nan_step,
            'last_optimizer_status': self.last_optimizer_status,
        }

    def restore_from_state(self, state):
        if not state:
            return
        self.lr_reductions_total = int(state.get(
            'lr_reductions_total', self.lr_reductions_total,
        ))

def diagnose_nan_loss(loss_dict):
    nan_components = []
    for k, v in loss_dict.items():
        if isinstance(v, (int, float)):
            val = v
        elif hasattr(v, 'item'):
            val = v.item()
        else:
            continue
        if math.isnan(val) or math.isinf(val):
            nan_components.append(f"{k}={val:.4f}")
    return nan_components

def _format_nan_batch_log(loss, loss_dict, nan_source, epoch, step):
    nan_components = diagnose_nan_loss(loss_dict)

    lines = [f"NaN/Inf loss at epoch {epoch}, step {step}"]
    lines.append(f"  Total: {loss.item() if hasattr(loss, 'item') else loss}")

    if nan_components:
        lines.append(f"  NaN sub-losses: {', '.join(nan_components)}")
    else:
        lines.append(f"  NaN sub-losses: NONE (all clean → aggregation bug?)")

    if nan_source:
        lines.append(f"  Hook first NaN at: {nan_source}")
    else:
        lines.append(f"  Hook first NaN at: no hook triggered (NaN from loss internals)")

    return '\n'.join(lines)
