"""
JiT DataModule for PyTorch Lightning
处理 ImageNet 数据集的加载、预处理和分布式采样
"""

import os
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import lightning.pytorch as pl

import numpy as np
from PIL import Image
# from util.crop import center_crop_arr

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class JiTDataModule(pl.LightningDataModule):
    """
    JiT 数据模块，负责 ImageNet 数据的加载和预处理
    
    Args:
        data_path: ImageNet 数据集的根路径
        img_size: 目标图像尺寸 (256 或 512)
        batch_size: 每个 GPU 的批次大小
        num_workers: 数据加载的工作进程数
        pin_memory: 是否将数据固定在内存中以加速 GPU 传输
        num_replicas: 分布式训练的副本数（GPU 数量）
        rank: 当前进程的 rank
    """
    
    def __init__(
        self,
        data_path: str = './data/imagenet',
        img_size: int = 256,
        batch_size: int = 128,
        num_workers: int = 12,
        pin_memory: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_replicas = num_replicas
        self.rank = rank
        
        # 数据集和采样器将在 setup 中初始化
        self.dataset_train = None
        self.sampler_train = None
        
    def prepare_data(self):
        """
        数据准备阶段（仅在主进程执行一次）
        这里可以添加下载数据、解压等操作
        """
        # ImageNet 数据集通常已经准备好，这里不需要额外操作
        pass
    
    def setup(self, stage: Optional[str] = None):
        """
        设置数据集和采样器
        
        Args:
            stage: 'fit', 'validate', 'test' 或 'predict'
        """
        if stage == 'fit' or stage is None:
            # 训练数据变换管道
            transform_train = self._get_train_transforms()
            
            # 加载 ImageNet 训练集
            train_path = os.path.join(self.data_path, 'train')
            self.dataset_train = datasets.ImageFolder(
                train_path, 
                transform=transform_train
            )
            
            # 设置分布式采样器
            if self.num_replicas is not None and self.rank is not None:
                self.sampler_train = DistributedSampler(
                    self.dataset_train,
                    num_replicas=self.num_replicas,
                    rank=self.rank,
                    shuffle=True
                )
            else:
                # 非分布式训练时使用默认采样
                self.sampler_train = None
            
            print(f"训练数据集: {self.dataset_train}")
            if self.sampler_train is not None:
                print(f"分布式采样器: {self.sampler_train}")
    
    def _get_train_transforms(self):
        """
        获取训练数据的变换管道
        
        Returns:
            transforms.Compose: 数据变换组合
        """
        transform_train = transforms.Compose([
            # 中心裁剪到目标尺寸
            transforms.Lambda(lambda img: center_crop_arr(img, self.img_size)),
            # 随机水平翻转（数据增强）
            transforms.RandomHorizontalFlip(),
            # 转换为 Tensor
            transforms.PILToTensor()
        ])
        return transform_train
    
    def train_dataloader(self):
        """
        创建训练数据加载器
        
        Returns:
            DataLoader: 训练数据加载器
        """
        if self.dataset_train is None:
            raise RuntimeError("请先调用 setup() 方法初始化数据集")
        
        return DataLoader(
            self.dataset_train,
            sampler=self.sampler_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # 丢弃最后一个不完整的 batch
            shuffle=(self.sampler_train is None),  # 只在非分布式时 shuffle
        )
    
    def val_dataloader(self):
        """
        创建验证数据加载器（可选）
        
        Returns:
            DataLoader: 验证数据加载器
        """
        # 如果需要验证集，可以在这里实现
        # 类似于 train_dataloader，但使用验证集路径和相应的变换
        pass
    
    def test_dataloader(self):
        """
        创建测试数据加载器（可选）
        
        Returns:
            DataLoader: 测试数据加载器
        """
        pass
    
    def teardown(self, stage: Optional[str] = None):
        """
        清理资源（在训练结束后调用）
        
        Args:
            stage: 'fit', 'validate', 'test' 或 'predict'
        """
        # 清理数据集和采样器
        self.dataset_train = None
        self.sampler_train = None
    
    def set_epoch(self, epoch: int):
        """
        设置当前 epoch（用于分布式训练的采样器）
        
        Args:
            epoch: 当前的 epoch 编号
        """
        if self.sampler_train is not None:
            self.sampler_train.set_epoch(epoch)
    
    def get_dataset_info(self):
        """
        获取数据集信息
        
        Returns:
            dict: 包含数据集信息的字典
        """
        if self.dataset_train is None:
            return {}
        
        return {
            'num_samples': len(self.dataset_train),
            'num_classes': len(self.dataset_train.classes),
            'img_size': self.img_size,
            'batch_size': self.batch_size,
        }


# 便捷函数：创建 JiT DataModule
def create_jit_datamodule(args):
    """
    从参数对象创建 JiT DataModule
    
    Args:
        args: 包含数据相关参数的对象
        
    Returns:
        JiTDataModule: 初始化好的数据模块
    """
    # 获取分布式训练参数（如果有的话）
    num_replicas = getattr(args, 'world_size', None)
    rank = getattr(args, 'rank', None)
    
    datamodule = JiTDataModule(
        data_path=args.data_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        num_replicas=num_replicas,
        rank=rank,
    )
    
    return datamodule

