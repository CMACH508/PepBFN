import wandb
import math
import torch
from core.utils.misc import BlackHole


def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2,)
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(train_cfg, optimizer):
    cfg = train_cfg.scheduler
    if cfg.type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg.max_iters,
            eta_min=cfg.min_lr,
        )
        def get_last_lr():
            return scheduler.get_last_lr()[0]
        return {
            'scheduler': scheduler,
            'interval': 'step',
        }, get_last_lr
    elif cfg.type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr
        )

        def get_last_lr():
            return optimizer.param_groups[0]['lr']
        
        return {
            'scheduler': scheduler,
            'monitor': 'val/recon_loss',
            'interval': 'epoch',
            'frequency': train_cfg.val_freq,
        }, get_last_lr
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)


def get_warmup_sched(cfg, optimizer):
    if cfg is None: return BlackHole()
    lambdas = [lambda it : (it / cfg.max_iters) if it <= cfg.max_iters else 1 for _ in optimizer.param_groups]
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
    return warmup_sched


def log_losses(loss, loss_dict, scalar_dict, it, tag, logger=BlackHole(), writer=BlackHole()):
    logstr = '[%s] Iter %05d' % (tag, it)
    logstr += ' | loss %.4f' % loss.item()
    for k, v in loss_dict.items():
        logstr += ' | loss(%s) %.4f' % (k, v.item())
    for k, v in scalar_dict.items():
        logstr += ' | %s %.4f' % (k, v.item() if isinstance(v, torch.Tensor) else v)
    logger.info(logstr)

    for k,v in loss_dict.items():
        wandb.log({f'train/loss_{k}': v}, step=it)
    for k,v in scalar_dict.items():
        wandb.log({f'train/{k}': v}, step=it)

    # writer.add_scalar('%s/loss' % tag, loss, it)
    # for k, v in loss_dict.items():
    #     writer.add_scalar('%s/loss_%s' % (tag, k), v, it)
    # for k, v in scalar_dict.items():
    #     writer.add_scalar('%s/%s' % (tag, k), v, it)
    # writer.flush()


class ScalarMetricAccumulator(object):

    def __init__(self):
        super().__init__()
        self.accum_dict = {}
        self.count_dict = {}

    @torch.no_grad()
    def add(self, name, value, batchsize=None, mode=None):
        assert mode is None or mode in ('mean', 'sum')

        if mode is None:
            delta = value.sum()
            count = value.size(0)
        elif mode == 'mean':
            delta = value * batchsize
            count = batchsize
        elif mode == 'sum':
            delta = value
            count = batchsize
        delta = delta.item() if isinstance(delta, torch.Tensor) else delta

        if name not in self.accum_dict:
            self.accum_dict[name] = 0
            self.count_dict[name] = 0
        self.accum_dict[name] += delta
        self.count_dict[name] += count

    def log(self, it, tag, logger=BlackHole(), writer=BlackHole()):
        summary = {k: self.accum_dict[k] / self.count_dict[k] for k in self.accum_dict}
        logstr = '[%s] Iter %05d' % (tag, it)
        for k, v in summary.items():
            logstr += ' | %s %.4f' % (k, v)
            writer.add_scalar('%s/%s' % (tag, k), v, it)
            wandb.log({f'{tag}/{k}': v}, step=it)
        logger.info(logstr)

    def get_average(self, name):
        return self.accum_dict[name] / self.count_dict[name]


def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}

    else:
        return obj


def sum_weighted_losses(losses, weights):
    """
    Args:
        losses:     Dict of scalar tensors.
        weights:    Dict of weights.
    """
    loss = 0
    for k in losses.keys():
        if weights is None:
            loss = loss + losses[k]
        else:
            loss = loss + weights[k] * losses[k]
    return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def process_dic(state_dict):
    new_state_dict = {}
    for k,v in state_dict.items():
        if 'module' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb