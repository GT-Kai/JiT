# JiT 配置文件说明

本目录包含了 JiT 项目的各种配置文件，使用 PyTorch Lightning CLI 的 YAML 配置格式。

## 📁 配置文件列表

### 主要配置文件

| 文件 | 模型 | 数据集 | 分辨率 | 说明 |
|------|------|--------|--------|------|
| `config.yaml` | JiT-B/16 | ImageNet | 256×256 | 默认配置，ImageNet 训练 |
| `config_jit_l16_256.yaml` | JiT-L/16 | ImageNet | 256×256 | Large 模型 |
| `config_jit_h16_256.yaml` | JiT-H/16 | ImageNet | 256×256 | Huge 模型 |
| `config_jit_b32_512.yaml` | JiT-B/32 | ImageNet | 512×512 | 高分辨率训练 |

### 测试配置文件

| 文件 | 模型 | 数据集 | 说明 |
|------|------|--------|------|
| `config_cifar10.yaml` | JiT-B/16 | CIFAR-10 | 快速测试，10 类 |
| `config_cifar100.yaml` | JiT-B/16 | CIFAR-100 | 快速测试，100 类 |

## 🚀 使用方法

### 基本用法

```bash
# 使用默认配置（ImageNet）
python main.py fit --config conf/config.yaml

# 使用 CIFAR-10 配置
python main.py fit --config conf/config_cifar10.yaml

# 使用 CIFAR-100 配置
python main.py fit --config conf/config_cifar100.yaml
```

### 命令行覆盖参数

```bash
# 修改 batch size
python main.py fit --config conf/config_cifar10.yaml --data.init_args.batch_size=32

# 修改学习率
python main.py fit --config conf/config_cifar10.yaml --model.init_args.learning_rate=0.0001

# 修改训练 epochs
python main.py fit --config conf/config_cifar10.yaml --trainer.max_epochs=200

# 使用多 GPU
python main.py fit --config conf/config.yaml --trainer.devices=4
```

## ⚙️ 配置文件结构

每个配置文件包含以下主要部分：

### 1. Trainer 配置

```yaml
trainer:
  max_epochs: 100              # 训练轮数
  devices: 1                   # GPU 数量
  accelerator: gpu             # 加速器类型
  strategy: auto               # 分布式策略
  precision: 16-mixed          # 混合精度训练
  log_every_n_steps: 50        # 日志记录频率
  
  # SwanLab 监控
  logger:
    class_path: swanlab.integration.pytorch_lightning.SwanLabLogger
    init_args:
      project: JiT-CIFAR10
      experiment_name: jit-b16-256
  
  # Callbacks
  callbacks:
    - class_path: callbacks.jit_callbacks.EMACallback
      init_args:
        ema_decay1: 0.9999
        ema_decay2: 0.9996
    # ... 其他 callbacks
```

### 2. Model 配置

```yaml
model:
  class_path: models.modelmodule.JiTLightningModule
  init_args:
    model_name: JiT-B/16        # 模型架构
    img_size: 256               # 图像尺寸
    num_classes: 10             # 类别数
    learning_rate: 0.00016      # 学习率
    weight_decay: 0.0           # 权重衰减
    ema_decay1: 0.9999          # EMA 衰减率 1
    ema_decay2: 0.9996          # EMA 衰减率 2
    P_mean: -0.8                # 时间步采样参数
    P_std: 0.8
    noise_scale: 1.0            # 噪声缩放
    t_eps: 0.05                 # 时间步 epsilon
    label_drop_prob: 0.1        # 标签丢弃概率（CFG）
    sampling_method: heun       # 采样方法
    num_sampling_steps: 50      # 采样步数
    cfg_scale: 2.9              # CFG 缩放因子
```

### 3. Data 配置

```yaml
data:
  class_path: datas.cifar_datamodule.CIFARDataModule
  init_args:
    dataset_name: cifar10       # 数据集名称
    data_path: ./data/cifar     # 数据路径
    download: true              # 自动下载
    img_size: 256               # 图像尺寸
    batch_size: 64              # 批次大小
    num_workers: 8              # 数据加载线程数
    pin_memory: true            # 固定内存
```

### 4. Callbacks 配置

```yaml
callbacks:
  # EMA 回调
  - class_path: callbacks.jit_callbacks.EMACallback
    init_args:
      ema_decay1: 0.9999
      ema_decay2: 0.9996
  
  # 模型检查点
  - class_path: callbacks.jit_callbacks.JiTModelCheckpoint
    init_args:
      dirpath: ./outputs/cifar10/checkpoints
      save_last_freq: 10
      save_milestone_freq: 50
  
  # 学习率调度
  - class_path: callbacks.jit_callbacks.LearningRateSchedulerCallback
    init_args:
      learning_rate: 0.00016
      lr_schedule: constant
      warmup_epochs: 5
      min_lr: 0.0
      epochs: 100
  
  # 指标记录
  - class_path: callbacks.jit_callbacks.MetricLoggerCallback
    init_args:
      log_freq: 50
```

## 📊 SwanLab 监控

所有配置文件都已集成 SwanLab 进行训练监控。详细使用方法请参考：

👉 [SwanLab 监控指南](../SWANLAB_GUIDE.md)

### 快速开始

```bash
# 1. 安装 SwanLab（已安装）
pip install swanlab

# 2. 运行训练（自动开始监控）
python main.py fit --config conf/config_cifar10.yaml

# 3. 在浏览器中查看实时监控
# SwanLab 会在终端输出链接
```

### 禁用 SwanLab

如果不想使用 SwanLab，可以在命令行中禁用：

```bash
python main.py fit --config conf/config_cifar10.yaml --trainer.logger=false
```

## 🎯 不同场景的推荐配置

### 场景 1: 快速测试代码

```bash
# 使用 CIFAR-10，小数据集，快速迭代
python main.py fit --config conf/config_cifar10.yaml \
  --trainer.max_epochs=10 \
  --trainer.limit_train_batches=100
```

### 场景 2: 完整 CIFAR 训练

```bash
# CIFAR-10 完整训练
python main.py fit --config conf/config_cifar10.yaml

# CIFAR-100 完整训练
python main.py fit --config conf/config_cifar100.yaml
```

### 场景 3: ImageNet 单 GPU 训练

```bash
python main.py fit --config conf/config.yaml \
  --trainer.devices=1 \
  --data.init_args.batch_size=64
```

### 场景 4: ImageNet 多 GPU 训练

```bash
# 4 GPU DDP 训练
python main.py fit --config conf/config.yaml \
  --trainer.devices=4 \
  --trainer.strategy=ddp \
  --data.init_args.batch_size=128
```

### 场景 5: 高分辨率训练

```bash
# 512×512 分辨率
python main.py fit --config conf/config_jit_b32_512.yaml \
  --trainer.devices=8 \
  --data.init_args.batch_size=64
```

## 🔧 自定义配置

### 创建新配置文件

1. 复制现有配置文件：
```bash
cp conf/config_cifar10.yaml conf/my_config.yaml
```

2. 修改参数：
```yaml
# 修改模型
model:
  init_args:
    model_name: JiT-L/16  # 改用 Large 模型

# 修改训练参数
trainer:
  max_epochs: 200
  devices: 2
```

3. 使用新配置：
```bash
python main.py fit --config conf/my_config.yaml
```

## 📝 参数说明

### 模型架构选项

| 模型名称 | 参数量 | Patch Size | 说明 |
|---------|--------|------------|------|
| `JiT-S/16` | ~22M | 16×16 | Small 模型 |
| `JiT-B/16` | ~86M | 16×16 | Base 模型（推荐） |
| `JiT-L/16` | ~307M | 16×16 | Large 模型 |
| `JiT-H/16` | ~632M | 16×16 | Huge 模型 |
| `JiT-B/32` | ~88M | 32×32 | 高分辨率专用 |

### 学习率调度选项

- `constant`: 恒定学习率（warmup 后）
- `cosine`: Cosine 衰减调度

### 采样方法选项

- `euler`: Euler 方法（快速）
- `heun`: Heun 方法（更准确，推荐）

### 分布式策略选项

- `auto`: 自动选择（推荐）
- `ddp`: DistributedDataParallel
- `ddp_spawn`: DDP with spawn
- `fsdp`: Fully Sharded Data Parallel（大模型）

## 🐛 故障排除

### 问题 1: CUDA Out of Memory

**解决方案：**
```bash
# 减小 batch size
python main.py fit --config conf/config.yaml --data.init_args.batch_size=32

# 或使用梯度累积
python main.py fit --config conf/config.yaml \
  --data.init_args.batch_size=32 \
  --trainer.accumulate_grad_batches=4
```

### 问题 2: 数据集未找到

**解决方案：**
```bash
# 对于 CIFAR，设置自动下载
python main.py fit --config conf/config_cifar10.yaml --data.init_args.download=true

# 对于 ImageNet，指定正确路径
python main.py fit --config conf/config.yaml --data.init_args.data_path=/path/to/imagenet
```

### 问题 3: SwanLab 连接问题

**解决方案：**
```bash
# 禁用 SwanLab
python main.py fit --config conf/config.yaml --trainer.logger=false

# 或使用本地模式（无需登录）
# SwanLab 默认就是本地模式，无需额外配置
```

## 📚 更多资源

- [数据模块文档](../datas/README.md)
- [模型模块文档](../models/README.md)
- [回调模块文档](../callbacks/README.md)
- [SwanLab 监控指南](../SWANLAB_GUIDE.md)

## 💡 最佳实践

1. **先用 CIFAR 测试**：在 ImageNet 上训练前，先用 CIFAR-10 验证代码
2. **使用混合精度**：`precision: 16-mixed` 可以加速训练并节省显存
3. **监控学习率**：确保学习率调度正常工作
4. **定期保存检查点**：设置合理的 `save_last_freq`
5. **使用 SwanLab**：实时监控训练过程，及时发现问题

## 🎉 开始训练

```bash
# 快速测试（CIFAR-10）
python main.py fit --config conf/config_cifar10.yaml

# 完整训练（ImageNet）
python main.py fit --config conf/config.yaml
```

祝训练顺利！🚀
