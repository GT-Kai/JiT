# JiT Lightning Callbacks

兼容原项目功能的 PyTorch Lightning Callbacks 模块。

## 📦 包含的 Callbacks

### 1. **EMACallback** - EMA 参数管理
- 维护两个版本的 EMA（Exponential Moving Average）参数
- 在每个训练批次后自动更新
- 支持切换到 EMA 参数进行评估

**使用示例：**
```python
from callbacks import EMACallback

ema_callback = EMACallback(
    ema_decay1=0.9999,  # 第一个 EMA 衰减率
    ema_decay2=0.9996,  # 第二个 EMA 衰减率
)
```

### 2. **JiTModelCheckpoint** - 模型检查点保存
- 定期保存模型、优化器和 EMA 参数
- 支持保存 last checkpoint 和里程碑 checkpoint
- 兼容原项目的 checkpoint 格式

**使用示例：**
```python
from callbacks import JiTModelCheckpoint

checkpoint_callback = JiTModelCheckpoint(
    dirpath='./checkpoints',
    save_last_freq=5,          # 每 5 个 epoch 保存一次 last
    save_milestone_freq=100,   # 每 100 个 epoch 保存一次里程碑
)
```

### 3. **FIDEvaluationCallback** - FID/IS 评估
- 定期生成图像并计算 FID 和 Inception Score
- 自动使用 EMA 参数进行评估
- 支持分布式评估

**使用示例：**
```python
from callbacks import FIDEvaluationCallback

fid_callback = FIDEvaluationCallback(
    eval_freq=40,              # 每 40 个 epoch 评估一次
    num_images=50000,          # 生成 50000 张图像
    batch_size=256,            # 评估批次大小
    num_classes=1000,          # 类别数量
    img_size=256,              # 图像尺寸
    use_ema=True,              # 使用 EMA 参数
)
```

### 4. **LearningRateSchedulerCallback** - 学习率调度
- 兼容原项目的学习率调度策略
- 支持 warmup 和多种调度方式
- 自动记录学习率到日志

**使用示例：**
```python
from callbacks import LearningRateSchedulerCallback

lr_callback = LearningRateSchedulerCallback(
    lr_schedule='constant',    # 学习率调度策略
    warmup_epochs=5,           # 预热轮数
    min_lr=0.0,                # 最小学习率
    epochs=600,                # 总轮数
)
```

### 5. **MetricLoggerCallback** - 指标记录
- 记录训练过程中的各种指标
- 兼容原项目的 MetricLogger
- 定期打印训练进度

**使用示例：**
```python
from callbacks import MetricLoggerCallback

metric_callback = MetricLoggerCallback(
    log_freq=100,  # 每 100 个 batch 打印一次
)
```

## 🚀 快速开始

### 方法 1: 使用默认配置

```python
from callbacks import create_default_callbacks
import lightning.pytorch as pl

# 创建所有默认 callbacks
callbacks = create_default_callbacks(
    ema_decay1=0.9999,
    ema_decay2=0.9996,
    save_dir='./checkpoints',
    eval_freq=40,
    img_size=256,
    epochs=600,
)

# 创建 Trainer
trainer = pl.Trainer(
    max_epochs=600,
    callbacks=callbacks,
    ...
)

# 训练
trainer.fit(model, datamodule)
```

### 方法 2: 自定义 Callbacks 组合

```python
from callbacks import (
    EMACallback,
    JiTModelCheckpoint,
    LearningRateSchedulerCallback,
)

callbacks = [
    EMACallback(ema_decay1=0.9999, ema_decay2=0.9996),
    JiTModelCheckpoint(dirpath='./checkpoints', save_last_freq=5),
    LearningRateSchedulerCallback(warmup_epochs=5, epochs=600),
]

trainer = pl.Trainer(callbacks=callbacks, ...)
```

### 方法 3: 从参数对象创建

```python
from callbacks import create_default_callbacks

# 从 argparse 参数创建
callbacks = create_default_callbacks(
    ema_decay1=args.ema_decay1,
    ema_decay2=args.ema_decay2,
    save_dir=args.output_dir,
    eval_freq=args.eval_freq,
    num_images=args.num_images,
    eval_batch_size=args.gen_bsz,
    num_classes=args.class_num,
    img_size=args.img_size,
    lr_schedule=args.lr_schedule,
    warmup_epochs=args.warmup_epochs,
    epochs=args.epochs,
    enable_fid_eval=args.online_eval,
)
```

## 📊 与原项目的对比

| 功能 | 原项目实现 | Lightning Callbacks |
|------|-----------|---------------------|
| **EMA 更新** | `engine_jit.py` 中手动调用 | `EMACallback` 自动处理 |
| **模型保存** | `util/misc.py` 中手动保存 | `JiTModelCheckpoint` 自动保存 |
| **FID 评估** | `engine_jit.py` 中手动评估 | `FIDEvaluationCallback` 自动评估 |
| **学习率调度** | `util/lr_sched.py` 手动调用 | `LearningRateSchedulerCallback` 自动调度 |
| **日志记录** | `util/misc.py` MetricLogger | `MetricLoggerCallback` 自动记录 |
| **代码行数** | ~300 行（分散在多个文件） | ~50 行（集中在一处） |

## 🎯 主要优势

1. **代码更简洁** - 从手动管理减少到自动处理
2. **更易维护** - 所有功能集中在 callbacks 中
3. **更好的错误处理** - Lightning 提供统一的错误处理
4. **自动分布式** - 无需手动处理分布式训练细节
5. **易于扩展** - 添加新功能只需创建新 callback
6. **统一接口** - 所有 callbacks 遵循相同的接口

## 📝 完整示例

```python
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from models.modelmodule import JiTLightningModule
from datas.datamodule import JiTDataModule
from callbacks import create_default_callbacks

# 创建模型
model = JiTLightningModule(
    model_name='JiT-B/16',
    img_size=256,
    num_classes=1000,
    learning_rate=1e-4,
)

# 创建数据模块
datamodule = JiTDataModule(
    data_path='./data/imagenet',
    img_size=256,
    batch_size=128,
)

# 创建 callbacks
callbacks = create_default_callbacks(
    ema_decay1=0.9999,
    ema_decay2=0.9996,
    save_dir='./outputs/jit_b16_256',
    eval_freq=40,
    img_size=256,
    epochs=600,
)

# 创建 logger
logger = TensorBoardLogger(save_dir='./logs', name='jit')

# 创建 Trainer
trainer = pl.Trainer(
    max_epochs=600,
    accelerator='gpu',
    devices=8,
    strategy='ddp',
    precision='bf16-mixed',
    callbacks=callbacks,
    logger=logger,
)

# 训练
trainer.fit(model, datamodule=datamodule)
```

## 🔧 高级用法

### 手动访问 EMA 参数

```python
# 获取 EMA callback
ema_callback = None
for cb in trainer.callbacks:
    if isinstance(cb, EMACallback):
        ema_callback = cb
        break

# 获取 EMA 状态字典
ema_state_dict = ema_callback.get_ema_state_dict(model, ema_version=1)

# 将 EMA 参数加载到模型
ema_callback.load_ema_to_model(model, ema_version=1)
```

### 自定义评估逻辑

```python
class CustomEvaluationCallback(FIDEvaluationCallback):
    def _evaluate(self, trainer, pl_module, epoch):
        # 自定义评估逻辑
        metrics = super()._evaluate(trainer, pl_module, epoch)
        
        # 添加额外的评估指标
        # ...
        
        return metrics
```

## 📚 更多示例

查看 `example_usage.py` 获取更多详细示例。

## ⚠️ 注意事项

1. **分布式训练** - Callbacks 已经处理了分布式训练的细节，无需手动同步
2. **EMA 参数** - EMA 参数会自动保存在 checkpoint 中
3. **FID 评估** - 需要安装 `torch-fidelity` 和准备 FID 统计文件
4. **内存管理** - FID 评估会自动清理 GPU 缓存

## 🐛 故障排除

### 问题 1: FID 评估失败
**解决方案**: 确保 `fid_stats` 目录存在且包含正确的统计文件

### 问题 2: Checkpoint 加载失败
**解决方案**: 确保 checkpoint 包含所有必要的键（model, optimizer, model_ema1, model_ema2）

### 问题 3: 分布式训练同步问题
**解决方案**: Callbacks 已经处理了 barrier，但确保使用正确的 strategy（如 'ddp'）

## 📄 许可证

与主项目相同的许可证。

