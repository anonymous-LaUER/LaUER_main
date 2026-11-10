"""Distributed feature gathering with autograd-aware gradient reduction."""
import torch
import torch.distributed as dist

def _gather_varlen(x):
    world_size = dist.get_world_size()
    B_local = torch.tensor([x.size(0)], device=x.device, dtype=torch.long)
    size_list = [torch.zeros(1, dtype=torch.long, device=x.device)
                 for _ in range(world_size)]
    dist.all_gather(size_list, B_local)
    sizes = [s.item() for s in size_list]
    max_size = max(sizes)

    if x.size(0) < max_size:
        pad = torch.zeros(max_size - x.size(0), *x.shape[1:],
                          device=x.device, dtype=x.dtype)
        x_padded = torch.cat([x, pad], dim=0)
    else:
        x_padded = x

    # all_gather padded tensors
    gathered = [torch.zeros(max_size, *x.shape[1:], device=x.device, dtype=x.dtype)
                for _ in range(world_size)]
    dist.all_gather(gathered, x_padded)

    # strip padding → cat
    output = torch.cat([g[:s] for g, s in zip(gathered, sizes)], dim=0)
    return output, sizes

class GatherLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        """x: [B_local, ...]"""
        output, sizes = _gather_varlen(x)
        ctx.sizes = sizes
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        rank = dist.get_rank()
        start = sum(ctx.sizes[:rank])
        end = start + ctx.sizes[rank]
        return grad_output[start:end]

def gather_features(x):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    return FixedGatherLayer.apply(x)

class FixedGatherLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        world_size = dist.get_world_size()
        gathered = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x)
        ctx.local_batch = x.size(0)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        rank = dist.get_rank()
        start = rank * ctx.local_batch
        return grad_output[start:start + ctx.local_batch]

@torch.no_grad()
def gather_features_nograd(x):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x

    output, _ = _gather_varlen(x)
    return output
