"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import wandb
import torch
import torch.nn as nn
import torch.utils.data

# ===== Added: 引入 torch.distributed，用于多卡训练时同步训练指标 =====
# 功能说明：
# 在 DDP 多 GPU 训练时，每张 GPU 只计算自己 batch 上的 intersection / union / target。
# 为了让 train_batch/mIoU、train_batch/mAcc、train_batch/allAcc 表示全局 batch 指标，
# 需要用 dist.all_reduce 把各 GPU 的统计量求和。
import torch.distributed as dist

from packaging import version
from functools import partial
from pathlib import Path

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry

# ===== Added: 引入 Pointcept 原本验证阶段使用的分割指标统计函数 =====
# 功能说明：
# intersection_and_union_gpu 用来根据 pred 和 label 计算：
# 1. 每个类别的 intersection
# 2. 每个类别的 union
# 3. 每个类别的 target
# 后续用这些统计量计算训练阶段的 mIoU / mAcc / allAcc。
from pointcept.utils.misc import intersection_and_union_gpu


TRAINERS = Registry("trainers")
AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.model = None
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)
        self._gradient_accumulation_counter = 0

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    # ===== Added: 计算当前模型梯度范数 =====
    # 功能说明：
    # 1. 用于记录 train_batch/grad_norm 和 train/grad_norm。
    # 2. 可以观察训练过程中梯度是否异常变大。
    # 3. 如果开启 clip_grad，可以结合 grad_norm_before_clip 和 grad_clipped 判断梯度裁剪是否频繁触发。
    @staticmethod
    def _compute_grad_norm(parameters, norm_type=2.0):
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return None

        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(p.grad.detach(), norm_type).to(device)
                    for p in parameters
                ]
            ),
            norm_type,
        )
        return total_norm

    # ===== Added: 训练阶段计算语义分割指标 =====
    # 功能说明：
    # 原始 Pointcept 训练阶段通常只记录 loss，不记录 train mIoU。
    # 这里新增训练 batch 上的：
    # 1. mIoU
    # 2. mAcc
    # 3. allAcc
    #
    # 注意：
    # 这个函数依赖模型 forward 返回 output_dict["seg_logits"]。
    # 所以 pointcept/models/default.py 的训练分支需要返回：
    # return dict(loss=loss, seg_logits=seg_logits.detach())
    def _add_train_seg_metrics(self, output_dict, input_dict):
        if "seg_logits" not in output_dict or "segment" not in input_dict:
            return

        with torch.no_grad():
            seg_logits = output_dict.pop("seg_logits")
            pred = seg_logits.max(1)[1]
            segment = input_dict["segment"]

            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.cfg.data.num_classes,
                self.cfg.data.ignore_index,
            )

            # ===== Added: 多 GPU 训练时同步各 GPU 的统计量 =====
            # 功能说明：
            # DDP 下每张卡只看到一部分 batch。
            # all_reduce 后再计算 mIoU / mAcc / allAcc，得到的是全局 train batch 指标。
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection)
                dist.all_reduce(union)
                dist.all_reduce(target)

            iou_class = intersection / (union + 1e-10)
            acc_class = intersection / (target + 1e-10)

            output_dict["mIoU"] = torch.mean(iou_class)
            output_dict["mAcc"] = torch.mean(acc_class)
            output_dict["allAcc"] = intersection.sum() / (target.sum() + 1e-10)

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        # Only clear gradients on first accumulation step
        if self._gradient_accumulation_counter == 0:
            self.optimizer.zero_grad()

        # Forward pass
        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            output_dict = self.model(input_dict)
            loss = (
                output_dict["loss"] / self.cfg.gradient_accumulation_steps
            )  # scale loss

        # ===== Added: 前向传播后计算训练阶段 mIoU / mAcc / allAcc =====
        # 功能说明：
        # 1. 这里使用 no_grad，不会参与反向传播。
        # 2. 会从 output_dict 中 pop 掉 seg_logits，避免日志系统长时间持有大 tensor。
        # 3. 计算结果会写回 output_dict，后续由 InformationWriter / EventStorage 写入日志和 TensorBoard。
        self._add_train_seg_metrics(output_dict, input_dict)

        # Backward pass
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        self._gradient_accumulation_counter += 1

        # ===== Added: 初始化梯度相关日志字段 =====
        # 功能说明：
        # 1. grad_norm：当前梯度范数。
        # 2. grad_norm_before_clip：梯度裁剪前范数，只有真正执行 optimizer.step 的时候才有实际意义。
        # 3. clip_grad_max_norm：当前配置的裁剪阈值。
        # 4. grad_clipped：是否触发梯度裁剪，1 表示触发，0 表示未触发。
        #
        # 注意：
        # 当使用 gradient_accumulation_steps > 1 时，并不是每个 mini-batch 都会执行 optimizer.step。
        # 所以未执行 optimizer.step 的累积步里，grad_norm_before_clip 和 grad_clipped 默认记为 0。
        grad_norm = self._compute_grad_norm(self.model.parameters())
        output_dict["grad_norm"] = (
            grad_norm.detach() if grad_norm is not None else loss.new_tensor(0.0)
        )
        output_dict["grad_norm_before_clip"] = loss.new_tensor(0.0)
        output_dict["clip_grad_max_norm"] = loss.new_tensor(
            float(self.cfg.clip_grad) if self.cfg.clip_grad is not None else 0.0
        )
        output_dict["grad_clipped"] = loss.new_tensor(0.0)

        # Perform optimizer step only when enough gradients have accumulated
        if self._gradient_accumulation_counter >= self.cfg.gradient_accumulation_steps:
            if self.cfg.enable_amp:
                self.scaler.unscale_(self.optimizer)

                # ===== Modified: AMP 分支增加梯度裁剪日志记录 =====
                # 功能说明：
                # 1. clip_grad_norm_ 的返回值是裁剪前的总梯度范数。
                # 2. 如果 grad_norm_before_clip > clip_grad_max_norm，说明本 step 触发了有效裁剪。
                # 3. 裁剪后重新计算 grad_norm，记录真正传入 optimizer.step 的梯度范数。
                if self.cfg.clip_grad is not None:
                    grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad
                    )
                    output_dict["grad_norm_before_clip"] = grad_norm_before_clip.detach()
                    output_dict["clip_grad_max_norm"] = loss.new_tensor(
                        float(self.cfg.clip_grad)
                    )
                    output_dict["grad_clipped"] = (
                        grad_norm_before_clip > self.cfg.clip_grad
                    ).float()

                    grad_norm_after_clip = self._compute_grad_norm(
                        self.model.parameters()
                    )
                    output_dict["grad_norm"] = (
                        grad_norm_after_clip.detach()
                        if grad_norm_after_clip is not None
                        else loss.new_tensor(0.0)
                    )

                self.scaler.step(self.optimizer)

                # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
                # Fix torch warning scheduler step before optimizer step.
                scale = self.scaler.get_scale()
                self.scaler.update()
                if scale <= self.scaler.get_scale():
                    self.scheduler.step()
            else:
                # ===== Modified: 非 AMP 分支增加梯度裁剪日志记录 =====
                # 功能说明：
                # 逻辑与 AMP 分支一致，只是不需要 scaler.unscale_。
                # 记录裁剪前梯度范数、裁剪阈值、是否触发裁剪，以及裁剪后的梯度范数。
                if self.cfg.clip_grad is not None:
                    grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad
                    )
                    output_dict["grad_norm_before_clip"] = grad_norm_before_clip.detach()
                    output_dict["clip_grad_max_norm"] = loss.new_tensor(
                        float(self.cfg.clip_grad)
                    )
                    output_dict["grad_clipped"] = (
                        grad_norm_before_clip > self.cfg.clip_grad
                    ).float()

                    grad_norm_after_clip = self._compute_grad_norm(
                        self.model.parameters()
                    )
                    output_dict["grad_norm"] = (
                        grad_norm_after_clip.detach()
                        if grad_norm_after_clip is not None
                        else loss.new_tensor(0.0)
                    )

                self.optimizer.step()
                self.scheduler.step()

            # Reset grad accumulation counter
            self._gradient_accumulation_counter = 0

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        if self.cfg.enable_wandb and comm.is_main_process():
            tag, name = Path(self.cfg.save_path).parts[-2:]
            wandb.init(
                project=self.cfg.wandb_project,
                name=f"{tag}/{name}",
                tags=[tag],
                dir=self.cfg.save_path,
                settings=wandb.Settings(api_key=self.cfg.wandb_key),
                config=self.cfg,
            )
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = (
            len(self.train_loader)
            * self.cfg.eval_epoch
            // self.cfg.gradient_accumulation_steps
        )
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            # deprecated warning
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader
