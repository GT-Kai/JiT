"""
JiT Model Module for PyTorch Lightning
基于 Lightning 架构的 JiT 扩散模型模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional

import lightning.pytorch as pl

from model_jit import JiT_models


class JiTLightningModule(pl.LightningModule):
    """
    JiT Lightning 模型模块，封装了训练、验证和生成逻辑
    
    Args:
        model_name: 模型名称 (如 'JiT-B/16', 'JiT-L/16', 'JiT-H/16' 等)
        img_size: 图像尺寸
        num_classes: 类别数量
        attn_dropout: 注意力 dropout 率
        proj_dropout: 投影层 dropout 率
        learning_rate: 学习率
        weight_decay: 权重衰减
        ema_decay1: 第一个 EMA 衰减率
        ema_decay2: 第二个 EMA 衰减率
        P_mean: 时间步采样分布均值
        P_std: 时间步采样分布标准差
        noise_scale: 噪声缩放因子
        t_eps: 时间步最小值（避免除零）
        label_drop_prob: 标签丢弃概率（用于 CFG）
        sampling_method: 采样方法 ('euler' 或 'heun')
        num_sampling_steps: 采样步数
        cfg_scale: Classifier-free guidance 缩放因子
        cfg_interval: CFG 应用区间 (min, max)
    """
    
    def __init__(
        self,
        model_name: str = 'JiT-B/16',
        img_size: int = 256,
        num_classes: int = 1000,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        ema_decay1: float = 0.9999,
        ema_decay2: float = 0.9996,
        P_mean: float = -0.8,
        P_std: float = 0.8,
        noise_scale: float = 1.0,
        t_eps: float = 5e-2,
        label_drop_prob: float = 0.1,
        sampling_method: str = 'heun',
        num_sampling_steps: int = 50,
        cfg_scale: float = 1.0,
        cfg_interval: tuple = (0.0, 1.0),
    ):
        super().__init__()
        
        # 保存超参数（会自动保存到 checkpoint）
        self.save_hyperparameters()
        
        # 创建 JiT 模型
        self.net = JiT_models[model_name](
            input_size=img_size,
            in_channels=3,
            num_classes=num_classes,
            attn_drop=attn_dropout,
            proj_drop=proj_dropout,
        )
        
        # 模型参数
        self.img_size = img_size
        self.num_classes = num_classes
        
        # 训练参数
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.label_drop_prob = label_drop_prob
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps
        self.noise_scale = noise_scale
        
        # EMA 参数
        self.ema_decay1 = ema_decay1
        self.ema_decay2 = ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None
        
        # 采样参数
        self.sampling_method = sampling_method
        self.num_sampling_steps = num_sampling_steps
        self.cfg_scale = cfg_scale
        self.cfg_interval = cfg_interval
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        模型前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
            t: 时间步 [B]
            y: 类别标签 [B]
            
        Returns:
            预测的图像 [B, C, H, W]
        """
        return self.net(x, t, y)
    
    def drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        随机丢弃标签（用于 Classifier-free Guidance）
        
        Args:
            labels: 类别标签 [B]
            
        Returns:
            处理后的标签 [B]
        """
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out
    
    def sample_timestep(self, n: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        采样时间步（使用 logit-normal 分布）
        
        Args:
            n: 批次大小
            device: 设备
            
        Returns:
            时间步 [n]
        """
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        训练步骤
        
        Args:
            batch: 数据批次 (images, labels)
            batch_idx: 批次索引
            
        Returns:
            损失值
        """
        images, labels = batch
        
        # 数据预处理：归一化到 [-1, 1]
        x = images.float() / 127.5 - 1.0
        
        # 随机丢弃标签（CFG）
        labels_dropped = self.drop_labels(labels)
        
        # 采样时间步
        t = self.sample_timestep(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        
        # 添加噪声
        e = torch.randn_like(x) * self.noise_scale
        
        # 扩散过程：z = t * x + (1 - t) * e
        z = t * x + (1 - t) * e
        
        # 计算目标速度场 v = (x - z) / (1 - t)
        v = (x - z) / (1 - t).clamp_min(self.t_eps)
        
        # 模型预测
        x_pred = self.forward(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        
        # L2 损失
        loss = F.mse_loss(v_pred, v, reduction='none')
        loss = loss.mean(dim=(1, 2, 3)).mean()
        
        # 记录日志
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        验证步骤
        
        Args:
            batch: 数据批次 (images, labels)
            batch_idx: 批次索引
            
        Returns:
            损失值
        """
        images, labels = batch
        
        # 数据预处理
        x = images.float() / 127.5 - 1.0
        
        # 采样时间步
        t = self.sample_timestep(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        
        # 添加噪声
        e = torch.randn_like(x) * self.noise_scale
        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)
        
        # 模型预测
        x_pred = self.forward(z, t.flatten(), labels)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        
        # 损失
        loss = F.mse_loss(v_pred, v, reduction='none')
        loss = loss.mean(dim=(1, 2, 3)).mean()
        
        # 记录日志
        self.log('val/loss', loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        
        Returns:
            优化器配置
        """
        # 为不同参数组设置不同的权重衰减
        no_decay = ['bias', 'norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.weight_decay,
            },
            {
                'params': [p for n, p in self.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.95)
        )
        
        return optimizer
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        每个训练批次结束后更新 EMA
        """
        if self.ema_params1 is None:
            # 首次初始化 EMA 参数
            self.ema_params1 = [p.detach().clone() for p in self.parameters()]
            self.ema_params2 = [p.detach().clone() for p in self.parameters()]
        else:
            # 更新 EMA
            self.update_ema()
    
    @torch.no_grad()
    def update_ema(self):
        """
        更新 Exponential Moving Average (EMA) 参数
        """
        source_params = list(self.parameters())
        
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
    
    @torch.no_grad()
    def generate(self, labels: torch.Tensor, use_ema: bool = True) -> torch.Tensor:
        """
        生成图像
        
        Args:
            labels: 类别标签 [B]
            use_ema: 是否使用 EMA 参数
            
        Returns:
            生成的图像 [B, C, H, W]，范围 [-1, 1]
        """
        device = labels.device
        bsz = labels.size(0)
        
        # 从噪声开始
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        
        # 创建时间步序列
        timesteps = torch.linspace(0.0, 1.0, self.num_sampling_steps + 1, device=device)
        timesteps = timesteps.view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)
        
        # 选择采样器
        if self.sampling_method == "euler":
            stepper = self._euler_step
        elif self.sampling_method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError(f"Unknown sampling method: {self.sampling_method}")
        
        # 如果使用 EMA，临时切换参数
        if use_ema and self.ema_params1 is not None:
            original_params = [p.detach().clone() for p in self.parameters()]
            for p, ema_p in zip(self.parameters(), self.ema_params1):
                p.data.copy_(ema_p)
        
        # ODE 采样
        for i in range(self.num_sampling_steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        
        # 最后一步使用 Euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        
        # 恢复原始参数
        if use_ema and self.ema_params1 is not None:
            for p, orig_p in zip(self.parameters(), original_params):
                p.data.copy_(orig_p)
        
        return z
    
    @torch.no_grad()
    def _forward_sample(self, z: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        带 Classifier-free Guidance 的采样前向传播
        
        Args:
            z: 当前状态 [B, C, H, W]
            t: 时间步 [B, 1, 1, 1]
            labels: 类别标签 [B]
            
        Returns:
            预测的速度场 [B, C, H, W]
        """
        # 条件预测
        x_cond = self.forward(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        
        # 无条件预测
        x_uncond = self.forward(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)
        
        # CFG 区间控制
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)
        
        # Classifier-free Guidance
        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)
    
    @torch.no_grad()
    def _euler_step(self, z: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, 
                    labels: torch.Tensor) -> torch.Tensor:
        """
        Euler 方法 ODE 求解步骤
        
        Args:
            z: 当前状态
            t: 当前时间步
            t_next: 下一个时间步
            labels: 类别标签
            
        Returns:
            下一个状态
        """
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next
    
    @torch.no_grad()
    def _heun_step(self, z: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, 
                   labels: torch.Tensor) -> torch.Tensor:
        """
        Heun 方法 ODE 求解步骤（二阶精度）
        
        Args:
            z: 当前状态
            t: 当前时间步
            t_next: 下一个时间步
            labels: 类别标签
            
        Returns:
            下一个状态
        """
        # 第一次预测
        v_pred_t = self._forward_sample(z, t, labels)
        
        # Euler 步骤
        z_next_euler = z + (t_next - t) * v_pred_t
        
        # 第二次预测
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)
        
        # Heun 校正
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        
        return z_next


# 便捷函数：从参数对象创建 Lightning 模型
def create_jit_lightning_module(args) -> JiTLightningModule:
    """
    从参数对象创建 JiT Lightning 模型
    
    Args:
        args: 包含模型相关参数的对象
        
    Returns:
        JiTLightningModule: 初始化好的 Lightning 模型
    """
    # 计算学习率（基于有效批次大小）
    if hasattr(args, 'lr') and args.lr is not None:
        learning_rate = args.lr
    elif hasattr(args, 'blr'):
        # 根据批次大小缩放学习率
        eff_batch_size = args.batch_size * getattr(args, 'world_size', 1)
        learning_rate = args.blr * eff_batch_size / 256
    else:
        learning_rate = 1e-4
    
    model = JiTLightningModule(
        model_name=args.model,
        img_size=args.img_size,
        num_classes=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
        ema_decay1=args.ema_decay1,
        ema_decay2=args.ema_decay2,
        P_mean=args.P_mean,
        P_std=args.P_std,
        noise_scale=args.noise_scale,
        t_eps=args.t_eps,
        label_drop_prob=args.label_drop_prob,
        sampling_method=args.sampling_method,
        num_sampling_steps=args.num_sampling_steps,
        cfg_scale=args.cfg,
        cfg_interval=(args.interval_min, args.interval_max),
    )
    
    return model

