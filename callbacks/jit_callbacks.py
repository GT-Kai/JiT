"""
JiT Lightning Callbacks
兼容原项目功能的 PyTorch Lightning Callbacks

实现功能：
1. EMA（Exponential Moving Average）更新和管理
2. 模型检查点保存（包含 EMA 参数）
3. FID/IS 评估
4. 学习率调度
5. 训练指标记录
"""

import os
import copy
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist
import numpy as np
import cv2

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from PIL import Image

import util.misc as misc
import util.lr_sched as lr_sched


class EMACallback(Callback):
    """
    Exponential Moving Average (EMA) Callback
    
    维护两个 EMA 版本的模型参数，在每个训练批次后更新
    """
    
    def __init__(
        self,
        ema_decay1: float = 0.9999,
        ema_decay2: float = 0.9996,
    ):
        super().__init__()
        self.ema_decay1 = ema_decay1
        self.ema_decay2 = ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None
    
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """训练开始时初始化 EMA 参数"""
        print("初始化 EMA 参数...")
        self.ema_params1 = [p.detach().clone() for p in pl_module.parameters()]
        self.ema_params2 = [p.detach().clone() for p in pl_module.parameters()]
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """每个训练批次结束后更新 EMA"""
        if self.ema_params1 is None:
            return
        
        source_params = list(pl_module.parameters())
        
        # 更新 EMA1
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        
        # 更新 EMA2
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
    
    def get_ema_state_dict(self, pl_module: pl.LightningModule, ema_version: int = 1) -> Dict:
        """
        获取 EMA 状态字典
        
        Args:
            pl_module: Lightning 模块
            ema_version: EMA 版本 (1 或 2)
        
        Returns:
            EMA 状态字典
        """
        if ema_version == 1:
            ema_params = self.ema_params1
        elif ema_version == 2:
            ema_params = self.ema_params2
        else:
            raise ValueError(f"Invalid ema_version: {ema_version}")
        
        if ema_params is None:
            return None
        
        ema_state_dict = copy.deepcopy(pl_module.state_dict())
        for i, (name, _) in enumerate(pl_module.named_parameters()):
            if name in ema_state_dict:
                ema_state_dict[name] = ema_params[i]
        
        return ema_state_dict
    
    def load_ema_to_model(self, pl_module: pl.LightningModule, ema_version: int = 1):
        """
        将 EMA 参数加载到模型中
        
        Args:
            pl_module: Lightning 模块
            ema_version: EMA 版本 (1 或 2)
        """
        ema_state_dict = self.get_ema_state_dict(pl_module, ema_version)
        if ema_state_dict is not None:
            pl_module.load_state_dict(ema_state_dict)


class JiTModelCheckpoint(Callback):
    """
    JiT 模型检查点保存 Callback
    
    保存模型、优化器、EMA 参数等
    兼容原项目的 checkpoint 格式
    """
    
    def __init__(
        self,
        dirpath: str = './checkpoints',
        save_last_freq: int = 5,
        save_milestone_freq: int = 100,
        save_top_k: int = 3,
        monitor: str = 'train/loss',
        mode: str = 'min',
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.save_last_freq = save_last_freq
        self.save_milestone_freq = save_milestone_freq
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.best_k_models = {}
        
        # 创建保存目录
        if misc.is_main_process():
            self.dirpath.mkdir(parents=True, exist_ok=True)
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """每个训练 epoch 结束时保存检查点"""
        epoch = trainer.current_epoch
        
        # 定期保存 last checkpoint
        if epoch % self.save_last_freq == 0 or epoch + 1 == trainer.max_epochs:
            self._save_checkpoint(trainer, pl_module, epoch, "last")
        
        # 里程碑 checkpoint
        if self.save_milestone_freq > 0 and epoch % self.save_milestone_freq == 0 and epoch > 0:
            self._save_checkpoint(trainer, pl_module, epoch, str(epoch))
    
    def _save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        epoch: int,
        epoch_name: str,
    ):
        """保存检查点"""
        if not misc.is_main_process():
            return
        
        checkpoint_path = self.dirpath / f'checkpoint-{epoch_name}.pth'
        
        # 基础信息
        to_save = {
            'model': pl_module.state_dict(),
            'epoch': epoch,
            'global_step': trainer.global_step,
        }
        
        # 保存优化器状态
        if trainer.optimizers:
            to_save['optimizer'] = trainer.optimizers[0].state_dict()
        
        # 保存 EMA 参数（如果存在）
        ema_callback = self._get_ema_callback(trainer)
        if ema_callback is not None:
            to_save['model_ema1'] = ema_callback.get_ema_state_dict(pl_module, ema_version=1)
            to_save['model_ema2'] = ema_callback.get_ema_state_dict(pl_module, ema_version=2)
        
        # 保存超参数
        to_save['hyper_parameters'] = pl_module.hparams
        
        torch.save(to_save, checkpoint_path)
        print(f"保存检查点到: {checkpoint_path}")
    
    def _get_ema_callback(self, trainer: pl.Trainer) -> Optional[EMACallback]:
        """获取 EMA Callback"""
        for callback in trainer.callbacks:
            if isinstance(callback, EMACallback):
                return callback
        return None


class FIDEvaluationCallback(Callback):
    """
    FID/IS 评估 Callback
    
    定期生成图像并计算 FID 和 Inception Score
    """
    
    def __init__(
        self,
        eval_freq: int = 40,
        num_images: int = 50000,
        batch_size: int = 256,
        num_classes: int = 1000,
        img_size: int = 256,
        output_dir: str = './outputs',
        fid_stats_file: Optional[str] = None,
        use_ema: bool = True,
    ):
        super().__init__()
        self.eval_freq = eval_freq
        self.num_images = num_images
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.img_size = img_size
        self.output_dir = output_dir
        self.fid_stats_file = fid_stats_file
        self.use_ema = use_ema
        
        # 自动选择 FID 统计文件
        if self.fid_stats_file is None:
            if img_size == 256:
                self.fid_stats_file = 'fid_stats/jit_in256_stats.npz'
            elif img_size == 512:
                self.fid_stats_file = 'fid_stats/jit_in512_stats.npz'
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """每个训练 epoch 结束时评估"""
        epoch = trainer.current_epoch
        
        # 检查是否需要评估
        if epoch % self.eval_freq != 0 and epoch + 1 != trainer.max_epochs:
            return
        
        print(f"\n开始评估 (Epoch {epoch})...")
        
        # 清空 GPU 缓存
        torch.cuda.empty_cache()
        
        # 执行评估
        try:
            metrics = self._evaluate(trainer, pl_module, epoch)
            
            # 记录指标
            if metrics is not None and misc.is_main_process():
                pl_module.log('eval/fid', metrics['fid'], sync_dist=True)
                pl_module.log('eval/is', metrics['is'], sync_dist=True)
                print(f"FID: {metrics['fid']:.4f}, IS: {metrics['is']:.4f}")
        
        except Exception as e:
            print(f"评估过程中出错: {e}")
        
        finally:
            # 清空 GPU 缓存
            torch.cuda.empty_cache()
    
    def _evaluate(self, trainer: pl.Trainer, pl_module: pl.LightningModule, epoch: int) -> Optional[Dict]:
        """执行评估"""
        pl_module.eval()
        
        world_size = misc.get_world_size()
        local_rank = misc.get_rank()
        num_steps = self.num_images // (self.batch_size * world_size) + 1
        
        # 构造保存文件夹名称
        cfg_scale = pl_module.hparams.get('cfg_scale', 1.0)
        cfg_interval = pl_module.hparams.get('cfg_interval', (0.0, 1.0))
        sampling_method = pl_module.hparams.get('sampling_method', 'heun')
        num_sampling_steps = pl_module.hparams.get('num_sampling_steps', 50)
        
        save_folder = os.path.join(
            self.output_dir,
            f"{sampling_method}-steps{num_sampling_steps}-cfg{cfg_scale}-"
            f"interval{cfg_interval[0]}-{cfg_interval[1]}-image{self.num_images}-res{self.img_size}"
        )
        
        if misc.is_main_process() and not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        if trainer.world_size > 1:
            dist.barrier()
        
        # 切换到 EMA 参数
        original_state_dict = None
        if self.use_ema:
            ema_callback = self._get_ema_callback(trainer)
            if ema_callback is not None:
                print("使用 EMA 参数进行评估")
                original_state_dict = copy.deepcopy(pl_module.state_dict())
                ema_callback.load_ema_to_model(pl_module, ema_version=1)
        
        # 生成图像
        class_label_gen_world = np.arange(0, self.num_classes).repeat(self.num_images // self.num_classes)
        class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
        
        for i in range(num_steps):
            print(f"生成步骤 {i}/{num_steps}")
            
            start_idx = world_size * self.batch_size * i + local_rank * self.batch_size
            end_idx = start_idx + self.batch_size
            labels_gen = class_label_gen_world[start_idx:end_idx]
            labels_gen = torch.Tensor(labels_gen).long().to(pl_module.device)
            
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                sampled_images = pl_module.generate(labels_gen, use_ema=False)  # 已经用 EMA 参数
            
            if trainer.world_size > 1:
                dist.barrier()
            
            # 反归一化图像
            sampled_images = (sampled_images + 1) / 2
            sampled_images = sampled_images.detach().cpu()
            
            # 分布式保存图像
            for b_id in range(sampled_images.size(0)):
                img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
                if img_id >= self.num_images:
                    break
                gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
                gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(os.path.join(save_folder, f'{str(img_id).zfill(5)}.png'), gen_img)
        
        if trainer.world_size > 1:
            dist.barrier()
        
        # 恢复原始参数
        if original_state_dict is not None:
            print("恢复原始参数")
            pl_module.load_state_dict(original_state_dict)
        
        # 计算 FID 和 IS
        metrics = None
        if misc.is_main_process() and self.fid_stats_file is not None:
            try:
                import torch_fidelity
                
                metrics_dict = torch_fidelity.calculate_metrics(
                    input1=save_folder,
                    input2=None,
                    fid_statistics_file=self.fid_stats_file,
                    cuda=True,
                    isc=True,
                    fid=True,
                    kid=False,
                    prc=False,
                    verbose=False,
                )
                
                metrics = {
                    'fid': metrics_dict['frechet_inception_distance'],
                    'is': metrics_dict['inception_score_mean'],
                }
                
                # 删除生成的图像文件夹
                shutil.rmtree(save_folder)
            
            except Exception as e:
                print(f"计算 FID/IS 时出错: {e}")
        
        if trainer.world_size > 1:
            dist.barrier()
        
        pl_module.train()
        return metrics
    
    def _get_ema_callback(self, trainer: pl.Trainer) -> Optional[EMACallback]:
        """获取 EMA Callback"""
        for callback in trainer.callbacks:
            if isinstance(callback, EMACallback):
                return callback
        return None


class LearningRateSchedulerCallback(Callback):
    """
    学习率调度 Callback
    
    兼容原项目的学习率调度策略
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-4,    # 基础学习率
        lr_schedule: str = 'constant',
        warmup_epochs: int = 5,
        min_lr: float = 0.0,
        epochs: int = 600,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.epochs = epochs
    
    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """每个训练批次开始时调整学习率"""
        if not trainer.optimizers:
            return
        
        optimizer = trainer.optimizers[0]
        epoch = trainer.current_epoch
        
        # 计算当前 epoch 的小数部分
        steps_per_epoch = len(trainer.train_dataloader)
        epoch_progress = batch_idx / steps_per_epoch
        current_epoch_float = epoch + epoch_progress
        
        # 调整学习率
        class Args:
            pass
        
        args = Args()
        # 从 pl_module 获取学习率，如果没有则使用回调的默认值
        if hasattr(pl_module, 'hparams') and hasattr(pl_module.hparams, 'learning_rate'):
            args.lr = pl_module.hparams.learning_rate
        else:
            args.lr = self.learning_rate
        args.lr_schedule = self.lr_schedule
        args.warmup_epochs = self.warmup_epochs
        args.min_lr = self.min_lr
        args.epochs = self.epochs
        
        lr_sched.adjust_learning_rate(optimizer, current_epoch_float, args)
        
        # 记录学习率
        if batch_idx % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            pl_module.log('train/lr', lr, prog_bar=True, on_step=True)


class ImageGenerationCallback(Callback):
    """
    图像生成和可视化 Callback
    
    在每个 epoch 结束后生成图像样本并上传到 SwanLab
    """
    
    def __init__(
        self,
        num_samples: int = 16,          # 每次生成的图像数量
        num_classes: int = 1000,        # 类别数
        img_size: int = 256,            # 图像尺寸
        use_ema: bool = True,           # 是否使用 EMA 参数
        ema_version: int = 1,           # 使用哪个 EMA (1 or 2)
        fixed_classes: Optional[list] = None,  # 固定的类别列表（用于可视化对比）
        log_every_n_epochs: int = 1,    # 每 N 个 epoch 记录一次
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        self.use_ema = use_ema
        self.ema_version = ema_version
        self.fixed_classes = fixed_classes
        self.log_every_n_epochs = log_every_n_epochs
        
        # 如果没有指定固定类别，随机选择一些
        if self.fixed_classes is None:
            # 选择一些有代表性的 ImageNet 类别
            # 或者随机选择
            import random
            random.seed(42)
            self.fixed_classes = random.sample(range(min(num_classes, 1000)), min(num_samples, 16))
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """每个训练 epoch 结束时生成图像"""
        current_epoch = trainer.current_epoch
        
        # 检查是否需要在这个 epoch 生成图像
        if (current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        
        # 只在主进程生成和记录
        if trainer.global_rank != 0:
            return
        
        print(f"\n{'='*50}")
        print(f"Epoch {current_epoch}: 生成测试图像...")
        print(f"{'='*50}")
        
        # 保存原始参数（如果使用 EMA）
        original_state_dict = None
        if self.use_ema:
            # 查找 EMA Callback
            ema_callback = None
            for callback in trainer.callbacks:
                if isinstance(callback, EMACallback):
                    ema_callback = callback
                    break
            
            if ema_callback is not None:
                original_state_dict = copy.deepcopy(pl_module.state_dict())
                ema_callback.load_ema_to_model(pl_module, ema_version=self.ema_version)
                print(f"✓ 使用 EMA-{self.ema_version} 参数生成")
            else:
                print("⚠️  未找到 EMA Callback，使用当前参数")
        
        # 设置为评估模式
        pl_module.eval()
        
        with torch.no_grad():
            # 准备类别标签
            if len(self.fixed_classes) >= self.num_samples:
                class_labels = torch.tensor(self.fixed_classes[:self.num_samples], device=pl_module.device)
            else:
                # 重复固定类别以达到 num_samples
                repeats = (self.num_samples + len(self.fixed_classes) - 1) // len(self.fixed_classes)
                class_labels = torch.tensor(self.fixed_classes * repeats, device=pl_module.device)[:self.num_samples]
            
            # 生成图像
            print(f"生成 {self.num_samples} 张图像，类别: {class_labels.tolist()}")
            
            # 已经手动加载了 EMA 参数（如果需要），所以这里不使用 generate 内部的 EMA
            generated_images = pl_module.generate(
                labels=class_labels,
                use_ema=False  # 不使用内部 EMA，因为我们已经手动加载了
            )
            
            # 转换为可视化格式 [0, 255]
            # generated_images: [B, C, H, W] -> 需要转换为 [B, H, W, C]
            generated_images = (generated_images + 1) / 2  # [-1, 1] -> [0, 1]
            generated_images = torch.clamp(generated_images * 255, 0, 255)
            generated_images = generated_images.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
            generated_images = generated_images.cpu().numpy().astype(np.uint8)
            
            # 创建图像网格
            grid_image = self._make_grid(generated_images, nrow=4)
            
            # 记录到 SwanLab
            if trainer.logger is not None:
                try:
                    # 检查是否是 SwanLab Logger
                    logger_name = type(trainer.logger).__name__
                    
                    if 'SwanLab' in logger_name:
                        import swanlab
                        # SwanLab 记录图像
                        trainer.logger.experiment.log({
                            # SwanLab 标签结尾不能有 '/'，用平铺名称
                            "generated_images": swanlab.Image(
                                grid_image,
                                caption=f"Epoch {current_epoch} - Classes: {class_labels.tolist()[:8]}..."
                            ),
                            "epoch": current_epoch
                        })
                        print(f"✓ 图像已上传到 SwanLab")
                    
                    elif 'TensorBoard' in logger_name:
                        # TensorBoard 记录图像
                        trainer.logger.experiment.add_image(
                            'generated_images',
                            grid_image.transpose(2, 0, 1),  # HWC -> CHW
                            global_step=current_epoch
                        )
                        print(f"✓ 图像已上传到 TensorBoard")
                    
                    else:
                        print(f"⚠️  未知的 Logger 类型: {logger_name}")
                
                except Exception as e:
                    print(f"⚠️  记录图像时出错: {e}")
            
            # 同时保存到本地
            save_dir = Path(trainer.default_root_dir) / "generated_samples"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"epoch_{current_epoch:04d}.png"
            
            Image.fromarray(grid_image).save(save_path)
            print(f"✓ 图像已保存到: {save_path}")
        
        # 恢复原始参数（如果使用了 EMA）
        if original_state_dict is not None:
            pl_module.load_state_dict(original_state_dict)
            print("✓ 已恢复原始参数")
        
        # 恢复训练模式
        pl_module.train()
        print(f"{'='*50}\n")
    
    def _make_grid(self, images: np.ndarray, nrow: int = 4) -> np.ndarray:
        """
        创建图像网格
        
        Args:
            images: (N, H, W, C) 的图像数组
            nrow: 每行的图像数量
        
        Returns:
            网格图像 (H_grid, W_grid, C)
        """
        n, h, w, c = images.shape
        ncol = (n + nrow - 1) // nrow
        
        # 创建网格画布
        grid_h = ncol * h
        grid_w = nrow * w
        grid = np.zeros((grid_h, grid_w, c), dtype=np.uint8)
        
        # 填充图像
        for idx, img in enumerate(images):
            row = idx // nrow
            col = idx % nrow
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = img
        
        return grid


class MetricLoggerCallback(Callback):
    """
    指标记录 Callback
    
    记录训练过程中的各种指标
    """
    
    def __init__(self, log_freq: int = 100):
        super().__init__()
        self.log_freq = log_freq
        self.metric_logger = misc.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """训练 epoch 开始时重置指标"""
        self.metric_logger = misc.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """训练批次结束时更新指标"""
        if outputs is not None and 'loss' in outputs:
            loss_value = outputs['loss'].item() if isinstance(outputs['loss'], torch.Tensor) else outputs['loss']
            self.metric_logger.update(loss=loss_value)
        
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]['lr']
            self.metric_logger.update(lr=lr)
        
        # 定期打印
        if batch_idx % self.log_freq == 0:
            print(f"Epoch: [{trainer.current_epoch}][{batch_idx}/{len(trainer.train_dataloader)}] {self.metric_logger}")


# 便捷函数：创建所有默认 callbacks
def create_default_callbacks(
    ema_decay1: float = 0.9999,
    ema_decay2: float = 0.9996,
    save_dir: str = './checkpoints',
    save_last_freq: int = 5,
    save_milestone_freq: int = 100,
    eval_freq: int = 40,
    num_images: int = 50000,
    eval_batch_size: int = 256,
    num_classes: int = 1000,
    img_size: int = 256,
    lr_schedule: str = 'constant',
    warmup_epochs: int = 5,
    min_lr: float = 0.0,
    epochs: int = 600,
    log_freq: int = 100,
    enable_fid_eval: bool = True,
):
    """
    创建所有默认的 JiT callbacks
    
    Args:
        ema_decay1: 第一个 EMA 衰减率
        ema_decay2: 第二个 EMA 衰减率
        save_dir: 检查点保存目录
        save_last_freq: 保存 last checkpoint 的频率
        save_milestone_freq: 保存里程碑 checkpoint 的频率
        eval_freq: 评估频率
        num_images: 评估时生成的图像数量
        eval_batch_size: 评估时的批次大小
        num_classes: 类别数量
        img_size: 图像尺寸
        lr_schedule: 学习率调度策略
        warmup_epochs: 预热轮数
        min_lr: 最小学习率
        epochs: 总轮数
        log_freq: 日志记录频率
        enable_fid_eval: 是否启用 FID 评估
    
    Returns:
        callbacks 列表
    """
    callbacks = []
    
    # EMA Callback
    callbacks.append(EMACallback(ema_decay1=ema_decay1, ema_decay2=ema_decay2))
    
    # Model Checkpoint Callback
    callbacks.append(JiTModelCheckpoint(
        dirpath=save_dir,
        save_last_freq=save_last_freq,
        save_milestone_freq=save_milestone_freq,
    ))
    
    # FID Evaluation Callback
    if enable_fid_eval:
        callbacks.append(FIDEvaluationCallback(
            eval_freq=eval_freq,
            num_images=num_images,
            batch_size=eval_batch_size,
            num_classes=num_classes,
            img_size=img_size,
            output_dir=save_dir,
        ))
    
    # Learning Rate Scheduler Callback
    callbacks.append(LearningRateSchedulerCallback(
        lr_schedule=lr_schedule,
        warmup_epochs=warmup_epochs,
        min_lr=min_lr,
        epochs=epochs,
    ))
    
    # Metric Logger Callback
    callbacks.append(MetricLoggerCallback(log_freq=log_freq))
    
    return callbacks

