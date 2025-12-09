# JiT Model Module

基于 PyTorch Lightning 的 JiT 扩散模型模块。

## 📦 模块组成

### `modelmodule.py` - Lightning 模型模块

包含 `JiTLightningModule` 类，封装了训练、验证和图像生成的完整逻辑。

## 🎯 核心类：JiTLightningModule

### 功能特点

- ✅ 继承自 `pl.LightningModule`
- ✅ 封装完整的扩散模型训练流程
- ✅ 支持 Classifier-free Guidance (CFG)
- ✅ 内置 EMA 参数管理
- ✅ 支持多种 ODE 采样方法（Euler、Heun）
- ✅ 自动保存和加载超参数
- ✅ 灵活的优化器配置

### 参数说明

```python
JiTLightningModule(
    # 模型架构参数
    model_name: str = 'JiT-B/16',           # 模型名称
    img_size: int = 256,                     # 图像尺寸
    num_classes: int = 1000,                 # 类别数量
    attn_dropout: float = 0.0,               # 注意力 dropout
    proj_dropout: float = 0.0,               # 投影层 dropout
    
    # 优化器参数
    learning_rate: float = 1e-4,             # 学习率
    weight_decay: float = 0.0,               # 权重衰减
    
    # EMA 参数
    ema_decay1: float = 0.9999,              # 第一个 EMA 衰减率
    ema_decay2: float = 0.9996,              # 第二个 EMA 衰减率
    
    # 扩散模型参数
    P_mean: float = -0.8,                    # 时间步采样均值
    P_std: float = 0.8,                      # 时间步采样标准差
    noise_scale: float = 1.0,                # 噪声缩放因子
    t_eps: float = 5e-2,                     # 时间步最小值
    label_drop_prob: float = 0.1,            # 标签丢弃概率（CFG）
    
    # 采样参数
    sampling_method: str = 'heun',           # 采样方法 ('euler' 或 'heun')
    num_sampling_steps: int = 50,            # 采样步数
    cfg_scale: float = 1.0,                  # CFG 缩放因子
    cfg_interval: tuple = (0.0, 1.0),        # CFG 应用区间
)
```

## 🏗️ 模型架构

### 可用的模型变体

| 模型名称 | 规模 | Patch Size | 参数量 | Hidden Size | Depth | Heads |
|---------|------|------------|--------|-------------|-------|-------|
| JiT-B/16 | Base | 16×16 | ~100M | 768 | 12 | 12 |
| JiT-B/32 | Base | 32×32 | ~100M | 768 | 12 | 12 |
| JiT-L/16 | Large | 16×16 | ~300M | 1024 | 24 | 16 |
| JiT-L/32 | Large | 32×32 | ~300M | 1024 | 24 | 16 |
| JiT-H/16 | Huge | 16×16 | ~600M | 1280 | 32 | 16 |
| JiT-H/32 | Huge | 32×32 | ~600M | 1280 | 32 | 16 |

### 模型组件

```
JiTLightningModule
├── net (JiT Transformer)
│   ├── x_embedder (BottleneckPatchEmbed)
│   ├── t_embedder (TimestepEmbedder)
│   ├── y_embedder (LabelEmbedder)
│   ├── blocks (JiTBlock × N)
│   │   ├── attn (Attention + RoPE)
│   │   ├── mlp (SwiGLU FFN)
│   │   └── adaLN (Adaptive Layer Norm)
│   └── final_layer (FinalLayer)
├── ema_params1 (EMA parameters)
└── ema_params2 (EMA parameters)
```

## 🚀 使用方法

### 方法 1: 直接创建模型

```python
from models.modelmodule import JiTLightningModule

# 创建模型
model = JiTLightningModule(
    model_name='JiT-B/16',
    img_size=256,
    num_classes=1000,
    learning_rate=1e-4,
    ema_decay1=0.9999,
    ema_decay2=0.9996,
    sampling_method='heun',
    num_sampling_steps=50,
    cfg_scale=2.9,
    cfg_interval=(0.1, 1.0),
)

# 查看模型信息
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
print(f"超参数: {model.hparams}")
```

### 方法 2: 与 Lightning Trainer 配合

```python
import lightning.pytorch as pl
from models.modelmodule import JiTLightningModule
from datas.datamodule import JiTDataModule

# 创建模型和数据模块
model = JiTLightningModule(model_name='JiT-B/16', img_size=256)
datamodule = JiTDataModule(data_path='./data/imagenet', img_size=256)

# 创建 Trainer
trainer = pl.Trainer(
    max_epochs=600,
    accelerator='gpu',
    devices=8,
    strategy='ddp',
    precision='bf16-mixed',
)

# 训练
trainer.fit(model, datamodule=datamodule)
```

### 方法 3: 从参数对象创建

```python
from models.modelmodule import create_jit_lightning_module

# 从 argparse 参数创建
model = create_jit_lightning_module(args)
```

### 方法 4: 加载预训练模型

```python
# 从 checkpoint 加载
model = JiTLightningModule.load_from_checkpoint('checkpoints/last.ckpt')

# 查看保存的超参数
print(model.hparams)
```

## 🔄 训练流程

### 前向传播

```python
def forward(self, x, t, y):
    """
    Args:
        x: 输入图像 [B, C, H, W]
        t: 时间步 [B]
        y: 类别标签 [B]
    
    Returns:
        预测的图像 [B, C, H, W]
    """
    return self.net(x, t, y)
```

### 训练步骤

```python
def training_step(self, batch, batch_idx):
    """
    训练步骤流程:
    1. 数据预处理 (归一化到 [-1, 1])
    2. 随机丢弃标签 (CFG)
    3. 采样时间步 (logit-normal 分布)
    4. 添加噪声
    5. 模型预测
    6. 计算 L2 损失
    7. 记录日志
    """
    images, labels = batch
    x = images.float() / 127.5 - 1.0
    
    labels_dropped = self.drop_labels(labels)
    t = self.sample_timestep(x.size(0), device=x.device)
    
    e = torch.randn_like(x) * self.noise_scale
    z = t * x + (1 - t) * e
    v = (x - z) / (1 - t).clamp_min(self.t_eps)
    
    x_pred = self(z, t.flatten(), labels_dropped)
    v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
    
    loss = F.mse_loss(v_pred, v)
    self.log('train/loss', loss)
    
    return loss
```

### 优化器配置

```python
def configure_optimizers(self):
    """
    配置 AdamW 优化器
    - 为 bias 和 norm 层设置零权重衰减
    - 使用 (0.9, 0.95) 的 beta 值
    """
    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=self.learning_rate,
        betas=(0.9, 0.95)
    )
    return optimizer
```

## 🎨 图像生成

### 基本生成

```python
# 创建模型
model = JiTLightningModule(...)
model.eval()
model.cuda()

# 准备标签
labels = torch.tensor([1, 2, 3, 4], device='cuda')  # 类别 ID

# 生成图像
with torch.no_grad():
    generated_images = model.generate(labels, use_ema=True)

# generated_images: [B, 3, H, W]，范围 [-1, 1]
```

### 高级生成选项

```python
# 使用不同的采样方法
model.sampling_method = 'euler'  # 或 'heun'
model.num_sampling_steps = 50    # 采样步数

# 调整 CFG 强度
model.cfg_scale = 2.9           # 引导强度
model.cfg_interval = (0.1, 1.0) # 应用区间

# 生成
images = model.generate(labels)

# 转换为 uint8 图像
images_uint8 = ((images + 1.0) * 127.5).clamp(0, 255).byte()
```

### ODE 采样方法

#### Euler 方法（一阶）
```python
def _euler_step(z, t, t_next, labels):
    """
    欧拉方法：
    z_next = z + (t_next - t) * v_pred
    """
    v_pred = _forward_sample(z, t, labels)
    z_next = z + (t_next - t) * v_pred
    return z_next
```

#### Heun 方法（二阶）
```python
def _heun_step(z, t, t_next, labels):
    """
    Heun 方法（改进的欧拉方法）：
    1. 预测 z_next (Euler)
    2. 计算修正后的速度
    3. 使用平均速度更新
    """
    v_pred_t = _forward_sample(z, t, labels)
    z_next_euler = z + (t_next - t) * v_pred_t
    v_pred_t_next = _forward_sample(z_next_euler, t_next, labels)
    v_pred = 0.5 * (v_pred_t + v_pred_t_next)
    z_next = z + (t_next - t) * v_pred
    return z_next
```

## 🔧 高级用法

### 自定义训练步骤

```python
class CustomJiTModule(JiTLightningModule):
    def training_step(self, batch, batch_idx):
        # 调用父类方法
        loss = super().training_step(batch, batch_idx)
        
        # 添加自定义逻辑
        if batch_idx % 100 == 0:
            # 记录额外的指标
            self.log('custom_metric', some_value)
        
        return loss
```

### 自定义采样方法

```python
class CustomJiTModule(JiTLightningModule):
    @torch.no_grad()
    def _custom_sampler(self, z, t, t_next, labels):
        """自定义 ODE 求解器"""
        # 实现你的采样逻辑
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred * some_factor
        return z_next
    
    def generate(self, labels, use_ema=True):
        # 使用自定义采样器
        for i in range(self.num_sampling_steps):
            z = self._custom_sampler(z, t, t_next, labels)
        return z
```

### 多 EMA 版本

```python
# 模型内置两个 EMA 版本
# EMA1: decay=0.9999 (更平滑)
# EMA2: decay=0.9996 (更快适应)

# 在 callbacks 中可以选择使用哪个版本
ema_callback.load_ema_to_model(model, ema_version=1)  # 使用 EMA1
# 或
ema_callback.load_ema_to_model(model, ema_version=2)  # 使用 EMA2
```

## 📊 性能优化

### 混合精度训练

```python
trainer = pl.Trainer(
    precision='bf16-mixed',  # 使用 BFloat16 混合精度
    # 或
    precision='16-mixed',    # 使用 Float16 混合精度
)
```

### 梯度累积

```python
trainer = pl.Trainer(
    accumulate_grad_batches=2,  # 累积 2 个批次的梯度
)
```

### 梯度裁剪

```python
trainer = pl.Trainer(
    gradient_clip_val=1.0,      # 梯度裁剪阈值
)
```

### 编译优化（PyTorch 2.0+）

```python
# 模型中的某些方法已经使用 @torch.compile 装饰
# 例如 JiTBlock.forward() 和 FinalLayer.forward()
# 这会自动进行 JIT 编译优化
```

## 📈 监控和日志

### 自动记录的指标

| 指标 | 说明 | 记录频率 |
|------|------|---------|
| `train/loss` | 训练损失 | 每个 step |
| `train/lr` | 学习率 | 每 100 steps |
| `val/loss` | 验证损失 | 每个 epoch |
| `eval/fid` | FID 分数 | 评估时 |
| `eval/is` | Inception Score | 评估时 |

### 自定义日志

```python
def training_step(self, batch, batch_idx):
    loss = ...
    
    # 记录到 TensorBoard
    self.log('train/loss', loss, prog_bar=True)
    self.log('train/custom_metric', value, on_step=True, on_epoch=True)
    
    return loss
```

## ⚙️ 配置建议

### JiT-B/16 @ 256×256

```python
model = JiTLightningModule(
    model_name='JiT-B/16',
    img_size=256,
    learning_rate=5e-5 * 8 / 2,  # 根据 GPU 数量缩放
    proj_dropout=0.0,
    P_mean=-0.8,
    P_std=0.8,
    noise_scale=1.0,
    cfg_scale=2.9,
)
```

### JiT-L/16 @ 256×256

```python
model = JiTLightningModule(
    model_name='JiT-L/16',
    img_size=256,
    learning_rate=5e-5 * 8 / 2,
    proj_dropout=0.0,
    cfg_scale=2.4,
)
```

### JiT-H/16 @ 256×256

```python
model = JiTLightningModule(
    model_name='JiT-H/16',
    img_size=256,
    learning_rate=5e-5 * 8 / 2,
    proj_dropout=0.2,  # 更大的模型需要更多正则化
    cfg_scale=2.2,
)
```

### 512×512 图像

```python
model = JiTLightningModule(
    model_name='JiT-B/32',  # 使用 32×32 patch
    img_size=512,
    noise_scale=2.0,        # 更大的噪声尺度
    cfg_scale=2.9,
)
```

## ⚠️ 注意事项

1. **图像归一化** - 训练时图像自动归一化到 [-1, 1]，生成时输出也是 [-1, 1]
2. **EMA 参数** - EMA 参数由 callbacks 管理，不要手动更新
3. **超参数保存** - `save_hyperparameters()` 会自动保存所有初始化参数到 checkpoint
4. **分布式训练** - 使用 DDP 时，模型会自动同步梯度
5. **内存使用** - Huge 模型需要至少 80GB GPU 内存（使用混合精度）

## 🐛 故障排除

### 问题 1: 训练损失不收敛
**可能原因**: 学习率过大或过小  
**解决方案**: 调整 `learning_rate`，建议范围 [1e-5, 1e-4]

### 问题 2: 生成图像质量差
**可能原因**: 
- 未使用 EMA 参数
- CFG 强度不合适
- 采样步数过少

**解决方案**:
```python
# 确保使用 EMA
images = model.generate(labels, use_ema=True)

# 调整 CFG
model.cfg_scale = 2.9
model.cfg_interval = (0.1, 1.0)

# 增加采样步数
model.num_sampling_steps = 100
```

### 问题 3: 内存溢出
**解决方案**:
- 减小 batch_size
- 使用混合精度训练
- 使用梯度累积
- 选择较小的模型（B 代替 L 或 H）

### 问题 4: 分布式训练速度慢
**解决方案**:
- 使用 `strategy='ddp'` 而不是 'ddp_spawn'
- 确保 `pin_memory=True` in DataModule
- 增加 `num_workers`

## 📚 相关文档

- [PyTorch Lightning LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)
- [JiT 论文](https://arxiv.org/abs/2511.13720)
- [Diffusion Models](https://arxiv.org/abs/2006.11239)

## 🔗 与其他模块的集成

### 与数据模块集成

```python
from models.modelmodule import JiTLightningModule
from datas.datamodule import JiTDataModule

# 确保图像尺寸匹配
img_size = 256
model = JiTLightningModule(img_size=img_size)
datamodule = JiTDataModule(img_size=img_size)
```

### 与 Callbacks 集成

```python
from models.modelmodule import JiTLightningModule
from callbacks import create_default_callbacks

model = JiTLightningModule(...)
callbacks = create_default_callbacks(
    ema_decay1=model.ema_decay1,
    ema_decay2=model.ema_decay2,
    img_size=model.img_size,
)
```

## 📄 许可证

与主项目相同的许可证。

