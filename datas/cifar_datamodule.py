"""
CIFAR DataModule for PyTorch Lightning
适配 CIFAR-10/CIFAR-100 数据集的数据模块
"""

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import lightning.pytorch as pl

import numpy as np
from PIL import Image


class CIFARDataModule(pl.LightningDataModule):
    """
    CIFAR 数据模块，支持 CIFAR-10 和 CIFAR-100
    
    Args:
        data_path: 数据集下载路径
        dataset_name: 'cifar10' 或 'cifar100'
        img_size: 目标图像尺寸 (会从 32x32 上采样)
        batch_size: 每个 GPU 的批次大小
        num_workers: 数据加载的工作进程数
        pin_memory: 是否将数据固定在内存中
        num_replicas: 分布式训练的副本数（GPU 数量）
        rank: 当前进程的 rank
        download: 是否自动下载数据集
    """
    
    def __init__(
        self,
        data_path: str = './data',
        dataset_name: str = 'cifar10',  # 'cifar10' or 'cifar100'
        img_size: int = 256,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        download: bool = True,
    ):
        super().__init__()
        self.data_path = data_path
        self.dataset_name = dataset_name.lower()
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_replicas = num_replicas
        self.rank = rank
        self.download = download
        
        # 数据集类别数
        self.num_classes = 10 if self.dataset_name == 'cifar10' else 100
        
        # 数据集和采样器将在 setup 中初始化
        self.dataset_train = None
        self.dataset_val = None
        self.sampler_train = None
        
    def prepare_data(self):
        """
        数据准备阶段（仅在主进程执行一次）
        自动下载 CIFAR 数据集
        """
        if self.download:
            if self.dataset_name == 'cifar10':
                datasets.CIFAR10(self.data_path, train=True, download=True)
                datasets.CIFAR10(self.data_path, train=False, download=True)
                print(f"✓ CIFAR-10 数据集已准备")
            elif self.dataset_name == 'cifar100':
                datasets.CIFAR100(self.data_path, train=True, download=True)
                datasets.CIFAR100(self.data_path, train=False, download=True)
                print(f"✓ CIFAR-100 数据集已准备")
    
    def setup(self, stage: Optional[str] = None):
        """
        设置数据集和采样器
        
        Args:
            stage: 'fit', 'validate', 'test' 或 'predict'
        """
        # 加载训练集
        if stage == 'fit' or stage is None:
            # 训练数据变换管道
            transform_train = self._get_train_transforms()
            
            # 加载 CIFAR 训练集
            if self.dataset_name == 'cifar10':
                self.dataset_train = datasets.CIFAR10(
                    self.data_path,
                    train=True,
                    transform=transform_train,
                    download=self.download
                )
            else:
                self.dataset_train = datasets.CIFAR100(
                    self.data_path,
                    train=True,
                    transform=transform_train,
                    download=self.download
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
                self.sampler_train = None
            
            print(f"✓ {self.dataset_name.upper()} 训练集加载完成")
            print(f"  - 样本数: {len(self.dataset_train)}")
            print(f"  - 类别数: {self.num_classes}")
        
        # 加载验证集/测试集（使用 CIFAR 的测试集作为验证集）
        if stage in ['fit', 'validate', 'test'] or stage is None:
            # 验证/测试数据变换管道
            transform_val = self._get_val_transforms()
            
            # 加载 CIFAR 测试集作为验证集
            if self.dataset_name == 'cifar10':
                self.dataset_val = datasets.CIFAR10(
                    self.data_path,
                    train=False,
                    transform=transform_val,
                    download=self.download
                )
            else:
                self.dataset_val = datasets.CIFAR100(
                    self.data_path,
                    train=False,
                    transform=transform_val,
                    download=self.download
                )
            
            print(f"✓ {self.dataset_name.upper()} 验证集加载完成")
            print(f"  - 样本数: {len(self.dataset_val)}")
    
    def _get_train_transforms(self):
        """
        获取训练数据的变换管道
        从 32x32 上采样到目标尺寸，并进行数据增强
        """
        transform_train = transforms.Compose([
            # 从 32x32 上采样到目标尺寸
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            # 数据增强
            transforms.RandomHorizontalFlip(),
            # 可选：添加更多增强
            # transforms.RandomCrop(self.img_size, padding=self.img_size // 8),
            # 转换为 Tensor
            transforms.ToTensor(),
            # 转换到 [0, 255] 范围（与 ImageNet DataModule 保持一致）
            transforms.Lambda(lambda x: x * 255),
        ])
        return transform_train
    
    def _get_val_transforms(self):
        """
        获取验证数据的变换管道
        """
        transform_val = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255),
        ])
        return transform_val
    
    def train_dataloader(self):
        """创建训练数据加载器"""
        if self.dataset_train is None:
            raise RuntimeError("请先调用 setup() 方法初始化数据集")
        
        return DataLoader(
            self.dataset_train,
            sampler=self.sampler_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            shuffle=(self.sampler_train is None),
        )
    
    def val_dataloader(self):
        """
        创建验证数据加载器
        使用 CIFAR 的测试集作为验证集
        """
        if self.dataset_val is None:
            raise RuntimeError("请先调用 setup() 方法初始化数据集")
        
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
    
    def test_dataloader(self):
        """创建测试数据加载器（使用验证集）"""
        return self.val_dataloader()
    
    def teardown(self, stage: Optional[str] = None):
        """清理资源"""
        self.dataset_train = None
        self.dataset_val = None
        self.sampler_train = None
    
    def set_epoch(self, epoch: int):
        """设置当前 epoch（用于分布式训练的采样器）"""
        if self.sampler_train is not None:
            self.sampler_train.set_epoch(epoch)
    
    def get_dataset_info(self):
        """获取数据集信息"""
        info = {
            'dataset_name': self.dataset_name,
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'batch_size': self.batch_size,
        }
        
        if self.dataset_train is not None:
            info['num_train_samples'] = len(self.dataset_train)
        if self.dataset_val is not None:
            info['num_val_samples'] = len(self.dataset_val)
        
        return info


# 便捷函数：创建 CIFAR DataModule
def create_cifar_datamodule(args, dataset_name='cifar10'):
    """
    从参数对象创建 CIFAR DataModule
    
    Args:
        args: 包含数据相关参数的对象
        dataset_name: 'cifar10' 或 'cifar100'
        
    Returns:
        CIFARDataModule: 初始化好的数据模块
    """
    # 获取分布式训练参数（如果有的话）
    num_replicas = getattr(args, 'world_size', None)
    rank = getattr(args, 'rank', None)
    
    datamodule = CIFARDataModule(
        data_path=getattr(args, 'data_path', './data'),
        dataset_name=dataset_name,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=getattr(args, 'num_workers', 4),
        pin_memory=getattr(args, 'pin_mem', True),
        num_replicas=num_replicas,
        rank=rank,
        download=True,
    )
    
    return datamodule

