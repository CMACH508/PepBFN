import tensorboard
import torch
import argparse
import os
import subprocess
from torch.utils.data import DataLoader, Subset
import datetime, pytz
from core.config.config import Config
from core.models.pep_train_loop import PepTrainLoop
from core.callbacks.basic import GradientClip, NormalizerCallback
from core.callbacks.ema import EMA
from core.callbacks.evaluate import EvalPep
from core.callbacks.construct import ConsPep

from core.dataset.pep_dataloader import PepDataset
from core.utils.data import PaddingCollate
from pytorch_lightning.strategies import DDPStrategy

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

from absl import logging
import torch.distributed as dist
from absl import logging as absl_logging
import logging as std_logging  # Python 标准库

# 1. 定义 rank‐zero 过滤器，只让 rank 0 输出
class RankZeroFilter(std_logging.Filter):
    def filter(self, record):
        # 未初始化（单卡）或是 rank0 时返回 True，其它 rank 返回 False
        return (not dist.is_available() 
                or not dist.is_initialized() 
                or dist.get_rank() == 0)

# 2. 拿到 absl 的底层 python handler，然后加上 filter
absl_handler = absl_logging.get_absl_handler().python_handler
absl_handler.addFilter(RankZeroFilter())
from pytorch_lightning.utilities import rank_zero_only

@rank_zero_only
def print_(message):
    print(message)
    
def get_dataloader(cfg, args):
    # Data
    logging.info('Loading datasets...')
    train_dataset = PepDataset(structure_dir=cfg.data.train.structure_dir, dataset_dir=cfg.data.train.dataset_dir,
                                            name=cfg.data.train.name, transform=None, reset=cfg.data.train.reset)
    val_dataset = PepDataset(structure_dir=cfg.data.val.structure_dir, dataset_dir=cfg.data.val.dataset_dir,
                                            name=cfg.data.val.name, transform=None, reset=cfg.data.val.reset)
    # train_dataset = Subset(train_dataset, range(16))
    # val_dataset = Subset(val_dataset, range(20))
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=PaddingCollate(), num_workers=cfg.train.num_workers, pin_memory=True)
    if "test_only" in args:
        # for testing, use batch_size=1
        # val_loader = DataLoader(Subset(val_dataset, [0,1,2]), batch_size=1, shuffle=False, collate_fn=PaddingCollate(eight=False), num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=PaddingCollate(eight=False), num_workers=1)
    else:
        val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=PaddingCollate(), num_workers=cfg.train.num_workers)
    logging.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))
    return train_loader, val_loader


def set_test_output_dir(cfg):
    path = cfg.accounting.test_outputs_dir
    version = 0
    while os.path.exists(path):
        version += 1
        path = cfg.accounting.test_outputs_dir + f'_v{version}'
    if version > 0:
        print_(f'{cfg.accounting.test_outputs_dir} already exists, change test_output_dir to {path}')
    else:
        print_(f'set test_output_dir as {path}')
    cfg.accounting.test_outputs_dir = path
    os.makedirs(cfg.accounting.test_outputs_dir, exist_ok=True)


def get_logger(cfg):
    logging_level = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "fatal": logging.FATAL,
    }
    logging.set_verbosity(logging_level[cfg.logging_level])
    if cfg.use_tblogger:
        os.makedirs(cfg.accounting.tb_logdir, exist_ok=True)
        logger = TensorBoardLogger(
            save_dir=cfg.accounting.tb_logdir,
            name=f"{cfg.exp_name}_{cfg.revision}"
                + f'_{datetime.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d-%H:%M:%S")}',
        )
    else:
        os.makedirs(cfg.accounting.wandb_logdir, exist_ok=True)
        if cfg.wandb_resume_id is not None:
            logger = WandbLogger(
                id=cfg.wandb_resume_id,
                project=cfg.project_name,
                offline=cfg.no_wandb,
                save_dir=cfg.accounting.wandb_logdir,
                resume='must',
            )
        else: # start a new run
            logger = WandbLogger(
                name=f"{cfg.exp_name}_{cfg.revision}"
                + f'_{datetime.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d-%H:%M:%S")}',
                project=cfg.project_name,
                offline=cfg.no_wandb,
                save_dir=cfg.accounting.wandb_logdir,
            )  # add wandb parameters
    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # test
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--test_ckpt_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--sample_steps", type=int, default=100)
    
    # ===============================================================
    # training
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--revision", type=str, default="debug")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb_resume_id", type=str, default=None)
    
    # global config
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--use_tblogger', type=bool, default=True)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--logging_level", type=str, default="warning")

    # train params
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['cosine', 'plateau'])
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=str, default='Q')  
    parser.add_argument("--pos_normalizer", type=float, default=5.0)    

    # bfn params
    parser.add_argument("--sigma1_coord", type=float, default=0.03)
    parser.add_argument("--lambda1_rot", type=float, default=10)
    parser.add_argument("--sigma1_angle", type=float, default=0.2)
    parser.add_argument("--beta1", type=float, default=1.2)
    parser.add_argument("--t_min", type=float, default=0.0001)
    parser.add_argument('--use_discrete_t', type=eval, default=True)
    parser.add_argument('--discrete_steps', type=int, default=1000)
    parser.add_argument('--destination_prediction', type=eval, default=True)
    parser.add_argument('--sampling_strategy', type=str, default='end_back_pmf', choices=['vanilla', 'end_back_pmf']) #vanilla or end_back
    _args = parser.parse_args()

    if _args.test_only and _args.test_ckpt_path is not None and _args.config_file is not None:
        if os.path.exists(_args.config_file):
            cfg = Config(_args.config_file)
            cfg.sample_steps = _args.sample_steps
            cfg.num_samples = _args.num_samples
            _args.test_ckpt_path = os.path.join(os.path.dirname(_args.config_file), "checkpoints", _args.test_ckpt_path)
        else:
            raise FileNotFoundError(f"Config file {_args.config_file} does not exist.")
    else:
        del _args.__dict__["test_only"]
        del _args.__dict__["test_ckpt_path"]
        del _args.__dict__["num_samples"]
        del _args.__dict__["sample_steps"]
        if not _args.resume:
            _args.__dict__["config_file"] = "configs/default.yaml"
            cfg = Config(**_args.__dict__)
        else:
            assert _args.config_file is not None, "Please provide a config file when resuming training."
            cfg = Config(**_args.__dict__)
            cfg.train.resume = True
        if not os.path.exists(cfg.accounting.logdir):
            os.makedirs(cfg.accounting.logdir, exist_ok=True)
        cfg.save2yaml(cfg.accounting.dump_config_path)
    
    seed_everything(cfg.seed, workers=True)
    logger = get_logger(cfg)
    
    train_loader, val_loader = get_dataloader(cfg, _args)
    logger.log_hyperparams(cfg.todict())
    print_(f"The config of this process is:\n{cfg}")

    model = PepTrainLoop(config=cfg)
    
    trainer = pl.Trainer(
        # accelerator='cpu',
        strategy=DDPStrategy(),
        default_root_dir=cfg.accounting.logdir,
        log_every_n_steps=1,
        max_epochs=cfg.train.epochs,
        check_val_every_n_epoch=cfg.train.val_freq,
        devices=[0],
        logger=logger,
        num_sanity_val_steps=0,
        inference_mode="test_only" in _args,
        callbacks=[
            GradientClip(max_grad_norm=cfg.train.max_grad_norm),  # time consuming
            NormalizerCallback(cfg.train.normalizer_dict),
            ModelCheckpoint(
                monitor="val/recon_loss",
                every_n_epochs=cfg.train.ckpt_freq,             
                dirpath=cfg.accounting.checkpoint_dir,
                filename="epoch{epoch:04d}_val_recon_loss{val/recon_loss:.2f}",
                save_top_k=-1,
                mode="min",
                auto_insert_metric_name=False,
                save_last=True,
            ),
            EMA(decay=cfg.train.ema_decay),
            ConsPep(cfg),
            # EvalPep(cfg)
        ],
    )
    if "test_only" not in _args:
        if cfg.train.resume:
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,ckpt_path='last')
        else:
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        trainer.test(model, dataloaders=val_loader, ckpt_path=_args.test_ckpt_path)
        # eval process
        # root_dir = os.path.dirname(_args.config_file)
        # log_path = os.path.join(root_dir, "eval.log")
        # cmd = f"bash train_eval.sh {root_dir} > {log_path} 2>&1"
        # subprocess.run(cmd, shell=True, check=True)
