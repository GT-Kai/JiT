# JiT Data Module

基于 PyTorch Lightning 的 JiT 数据加载和预处理模块。

## 📦 模块组成

### `datamodule.py` - 核心数据模块

包含 `JiTDataModule` 类，负责 ImageNet 数据的加载、预处理和分布式采样。

## 🎯 核心类：JiTDataModule

### 功能特点

- ✅ 继承自 `pl.LightningDataModule`
- ✅ 自动处理分布式训练的数据采样
- ✅ 内置图像预处理流程
- ✅ 支持自定义数据增强
- ✅ 灵活的配置选项

### 参数说明

```python
JiTDataModule(
    data_path: str = './data/imagenet',     # ImageNet 数据集路径
    img_size: int = 256,                     # 图像尺寸 (256 或 512)
    batch_size: int = 128,                   # 每个 GPU 的批次大小
    num_workers: int = 12,                   # 数据加载的工作进程数
    pin_memory: bool = True,                 # 是否固定内存
    num_replicas: Optional[int] = None,      # 分布式副本数（GPU 数量）
    rank: Optional[int] = None,              # 当前进程的 rank
)
```

### 主要方法

| 方法 | 说明 |
|------|------|
| `prepare_data()` | 数据准备（下载、解压等，仅主进程执行一次） |
| `setup(stage)` | 设置数据集和采样器 |
| `train_dataloader()` | 返回训练数据加载器 |
| `val_dataloader()` | 返回验证数据加载器（可选） |
| `set_epoch(epoch)` | 设置当前 epoch（用于分布式采样） |
| `get_dataset_info()` | 获取数据集信息 |

## 🔄 数据预处理流程

### 1. 中心裁剪（Center Crop）

使用 ADM (OpenAI) 的中心裁剪实现：
- 自动缩放图像到目标尺寸
- 保持图像长宽比
- 裁剪中心区域

```python
def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM
    """
    # 1. 逐步缩小（使用 BOX 采样）
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), 
            resample=Image.BOX
        )
    
    # 2. 缩放到目标尺寸（使用 BICUBIC 插值）
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), 
        resample=Image.BICUBIC
    )
    
    # 3. 裁剪中心区域
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
```

### 2. 数据增强

默认的数据增强流程：
```python
transform_train = transforms.Compose([
    transforms.Lambda(lambda img: center_crop_arr(img, img_size)),  # 中心裁剪
    transforms.RandomHorizontalFlip(),                               # 随机水平翻转
    transforms.PILToTensor()                                         # 转换为 Tensor
])
```

## 🚀 使用方法

### 方法 1: 直接创建

```python
from datas.datamodule import JiTDataModule

# 创建数据模块
datamodule = JiTDataModule(
    data_path='./data/imagenet',
    img_size=256,
    batch_size=128,
    num_workers=12,
    pin_memory=True,
)

# 设置数据集
datamodule.setup(stage='fit')

# 获取数据加载器
train_loader = datamodule.train_dataloader()

# 查看数据集信息
info = datamodule.get_dataset_info()
print(f"样本数: {info['num_samples']}")
print(f"类别数: {info['num_classes']}")
```

### 方法 2: 与 Lightning Trainer 配合

```python
import lightning.pytorch as pl
from datas.datamodule import JiTDataModule
from models.modelmodule import JiTLightningModule

# 创建数据模块
datamodule = JiTDataModule(
    data_path='./data/imagenet',
    img_size=256,
    batch_size=128,
)

# 创建模型
model = JiTLightningModule(...)

# 创建 Trainer
trainer = pl.Trainer(
    max_epochs=600,
    accelerator='gpu',
    devices=8,
)

# 训练（数据加载自动处理）
trainer.fit(model, datamodule=datamodule)
```

### 方法 3: 分布式训练

```python
from datas.datamodule import JiTDataModule

# 在分布式训练中使用
datamodule = JiTDataModule(
    data_path='./data/imagenet',
    img_size=256,
    batch_size=128,
    num_replicas=8,      # 8 个 GPU
    rank=local_rank,     # 当前进程的 rank
)

# Lightning Trainer 会自动处理分布式细节
trainer = pl.Trainer(
    devices=8,
    strategy='ddp',      # 分布式数据并行
)

# 在训练循环中设置 epoch（用于正确的数据 shuffle）
for epoch in range(epochs):
    datamodule.set_epoch(epoch)
    # 训练...
```

### 方法 4: 从参数对象创建

```python
from datas.datamodule import create_jit_datamodule

# 从 args 对象创建
datamodule = create_jit_datamodule(args)
```

## 📁 数据集结构

期望的 ImageNet 数据集目录结构：

```
data/imagenet/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   └── ...
│   ├── n01443537/
│   │   └── ...
│   └── ...
└── val/ (可选)
    ├── n01440764/
    └── ...
```

## 🔧 高级用法

### 自定义数据增强

```python
class CustomJiTDataModule(JiTDataModule):
    def _get_train_transforms(self):
        """自定义训练数据变换"""
        return transforms.Compose([
            transforms.Lambda(lambda img: center_crop_arr(img, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),  # 添加颜色抖动
            transforms.PILToTensor()
        ])
```

### 添加验证集

```python
class JiTDataModuleWithVal(JiTDataModule):
    def setup(self, stage=None):
        super().setup(stage)
        
        if stage == 'fit' or stage == 'validate':
            # 加载验证集
            val_path = os.path.join(self.data_path, 'val')
            transform_val = transforms.Compose([
                transforms.Lambda(lambda img: center_crop_arr(img, self.img_size)),
                transforms.PILToTensor()
            ])
            self.dataset_val = datasets.ImageFolder(val_path, transform=transform_val)
    
    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
```

### 动态批次大小

```python
# 根据图像尺寸调整批次大小
img_size = 512
batch_size = 128 if img_size == 256 else 64

datamodule = JiTDataModule(
    data_path='./data/imagenet',
    img_size=img_size,
    batch_size=batch_size,
)
```

## 📊 数据统计

### ImageNet-1K 数据集

- **训练集**: 1,281,167 张图像
- **验证集**: 50,000 张图像
- **类别数**: 1,000 类
- **图像格式**: JPEG
- **图像尺寸**: 可变（经过预处理统一到 256×256 或 512×512）

### 内存和性能

| 配置 | 批次大小 | 工作进程 | GPU 内存占用 | 加载速度 |
|------|---------|---------|-------------|---------|
| 256×256 | 128 | 12 | ~24 GB | ~1000 img/s |
| 512×512 | 64 | 12 | ~40 GB | ~500 img/s |

## ⚠️ 注意事项

1. **数据路径** - 确保 `data_path` 指向正确的 ImageNet 目录
2. **内存固定** - `pin_memory=True` 可以加速 GPU 传输，但会占用更多 CPU 内存
3. **工作进程** - `num_workers` 应根据 CPU 核心数调整（通常设置为 8-16）
4. **分布式训练** - 使用 Lightning Trainer 时，`num_replicas` 和 `rank` 会自动设置
5. **Epoch 设置** - 在分布式训练中，必须在每个 epoch 调用 `set_epoch()` 以正确 shuffle 数据

## 🐛 故障排除

### 问题 1: 数据加载缓慢
**原因**: `num_workers` 设置过小  
**解决方案**: 增加 `num_workers` 到 12-16

### 问题 2: 内存不足
**原因**: `pin_memory=True` 占用过多 CPU 内存  
**解决方案**: 设置 `pin_memory=False`

### 问题 3: 分布式训练数据重复
**原因**: 未正确设置 epoch  
**解决方案**: 在每个 epoch 调用 `datamodule.set_epoch(epoch)`

### 问题 4: 图像尺寸不匹配
**原因**: 数据集中包含损坏的图像  
**解决方案**: 检查并清理数据集

## 📚 相关文档

- [PyTorch Lightning DataModule 文档](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)
- [ImageNet 数据集](https://www.image-net.org/)
- [torchvision.datasets.ImageFolder](https://pytorch.org/vision/stable/datasets.html#imagefolder)

## 🔗 与其他模块的集成

### 与模型模块集成

```python
from datas.datamodule import JiTDataModule
from models.modelmodule import JiTLightningModule

datamodule = JiTDataModule(img_size=256, batch_size=128)
model = JiTLightningModule(img_size=256)

# 图像尺寸必须匹配
assert datamodule.img_size == model.img_size
```

### 与 Callbacks 集成

```python
from datas.datamodule import JiTDataModule
from callbacks import create_default_callbacks

datamodule = JiTDataModule(...)
callbacks = create_default_callbacks(
    img_size=datamodule.img_size,
    num_classes=1000,
)
```

## 📄 许可证

与主项目相同的许可证。

