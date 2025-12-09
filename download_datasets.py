"""
数据集预下载脚本

提前下载和准备数据集到指定路径

用法:
    # 下载 CIFAR-10
    python download_datasets.py --dataset cifar10 --path ./data/cifar
    python download_datasets.py --dataset cifar10 --path /fs/hdd/share/lick/datasets/cifar10
    
    # 下载 CIFAR-100
    python download_datasets.py --dataset cifar100 --path ./data/cifar
    python download_datasets.py --dataset cifar100 --path /fs/hdd/share/lick/datasets/cifar100
    
    # 下载所有 CIFAR 数据集
    python download_datasets.py --dataset all_cifar --path ./data/cifar
    python download_datasets.py --dataset all_cifar --path /fs/hdd/share/lick/datasets/cifar100
    
    # 验证已下载的数据集
    python download_datasets.py --dataset cifar10 --path ./data/cifar --verify
    python download_datasets.py --dataset cifar10 --path /fs/hdd/share/lick/datasets/cifar10 --verify
    python download_datasets.py --dataset cifar100 --path /fs/hdd/share/lick/datasets/cifar100 --verify
    python download_datasets.py --dataset all_cifar --path /fs/hdd/share/lick/datasets/cifar100 --verify
    
    # 列出已下载的数据集
    python download_datasets.py --list --path ./data/cifar
    python download_datasets.py --list --path /fs/hdd/share/lick/datasets/cifar10
    python download_datasets.py --list --path /fs/hdd/share/lick/datasets/cifar100
    
    # 显示数据集信息
    python download_datasets.py --dataset cifar10 --path ./data/cifar --info
    python download_datasets.py --dataset cifar10 --path /fs/hdd/share/lick/datasets/cifar10 --info
    python download_datasets.py --dataset cifar100 --path /fs/hdd/share/lick/datasets/cifar100 --info
    python download_datasets.py --dataset all_cifar --path /fs/hdd/share/lick/datasets/cifar100 --info
    
"""

import argparse
import os
from pathlib import Path
import torchvision.datasets as datasets
from torchvision import transforms


def download_cifar10(data_path, verify=False):
    """
    下载 CIFAR-10 数据集
    
    Args:
        data_path: 数据保存路径
        verify: 是否验证数据集
    """
    print("="*60)
    print("下载 CIFAR-10 数据集")
    print("="*60)
    
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n下载路径: {data_path.absolute()}")
    print("开始下载...\n")
    
    # 下载训练集
    print("1. 下载训练集...")
    train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=None
    )
    print(f"   ✓ 训练集下载完成: {len(train_dataset)} 张图像")
    
    # 下载测试集
    print("\n2. 下载测试集...")
    test_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=None
    )
    print(f"   ✓ 测试集下载完成: {len(test_dataset)} 张图像")
    
    # 验证数据集
    if verify:
        print("\n3. 验证数据集...")
        verify_dataset(train_dataset, "CIFAR-10 训练集")
        verify_dataset(test_dataset, "CIFAR-10 测试集")
    
    print(f"\n✓ CIFAR-10 数据集准备完成！")
    print(f"  总样本数: {len(train_dataset) + len(test_dataset)}")
    print(f"  类别数: 10")
    print(f"  存储路径: {data_path.absolute()}")
    
    return train_dataset, test_dataset


def download_cifar100(data_path, verify=False):
    """
    下载 CIFAR-100 数据集
    
    Args:
        data_path: 数据保存路径
        verify: 是否验证数据集
    """
    print("="*60)
    print("下载 CIFAR-100 数据集")
    print("="*60)
    
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n下载路径: {data_path.absolute()}")
    print("开始下载...\n")
    
    # 下载训练集
    print("1. 下载训练集...")
    train_dataset = datasets.CIFAR100(
        root=data_path,
        train=True,
        download=True,
        transform=None
    )
    print(f"   ✓ 训练集下载完成: {len(train_dataset)} 张图像")
    
    # 下载测试集
    print("\n2. 下载测试集...")
    test_dataset = datasets.CIFAR100(
        root=data_path,
        train=False,
        download=True,
        transform=None
    )
    print(f"   ✓ 测试集下载完成: {len(test_dataset)} 张图像")
    
    # 验证数据集
    if verify:
        print("\n3. 验证数据集...")
        verify_dataset(train_dataset, "CIFAR-100 训练集")
        verify_dataset(test_dataset, "CIFAR-100 测试集")
    
    print(f"\n✓ CIFAR-100 数据集准备完成！")
    print(f"  总样本数: {len(train_dataset) + len(test_dataset)}")
    print(f"  类别数: 100")
    print(f"  存储路径: {data_path.absolute()}")
    
    return train_dataset, test_dataset


def download_all_cifar(data_path, verify=False):
    """下载所有 CIFAR 数据集"""
    print("="*60)
    print("下载所有 CIFAR 数据集")
    print("="*60)
    
    # 下载 CIFAR-10
    download_cifar10(data_path, verify)
    
    print("\n" + "-"*60 + "\n")
    
    # 下载 CIFAR-100
    download_cifar100(data_path, verify)


def verify_dataset(dataset, name):
    """
    验证数据集
    
    Args:
        dataset: 数据集对象
        name: 数据集名称
    """
    print(f"\n   验证 {name}...")
    
    try:
        # 尝试加载第一个样本
        img, label = dataset[0]
        print(f"   ✓ 数据集可正常访问")
        print(f"     - 图像尺寸: {img.size}")
        print(f"     - 标签范围: 0-{len(dataset.classes)-1}")
        print(f"     - 样本数: {len(dataset)}")
        
        # 检查前 100 个样本
        print(f"   检查前 100 个样本...")
        for i in range(min(100, len(dataset))):
            _ = dataset[i]
        print(f"   ✓ 前 100 个样本正常")
        
    except Exception as e:
        print(f"   ✗ 数据集验证失败: {e}")
        raise


def check_disk_space(path, required_mb=500):
    """
    检查磁盘空间
    
    Args:
        path: 检查路径
        required_mb: 所需空间（MB）
    """
    import shutil
    
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    required_gb = required_mb / 1024
    
    print(f"\n磁盘空间检查:")
    print(f"  路径: {path}")
    print(f"  可用空间: {free_gb:.2f} GB")
    print(f"  所需空间: {required_gb:.2f} GB")
    
    if free_gb < required_gb:
        print(f"  ⚠️  警告: 磁盘空间可能不足！")
        return False
    else:
        print(f"  ✓ 磁盘空间充足")
        return True


def show_download_info(dataset_name):
    """显示数据集下载信息"""
    info = {
        'cifar10': {
            'name': 'CIFAR-10',
            'size': '~170 MB',
            'samples': '60,000',
            'classes': 10,
            'url': 'https://www.cs.toronto.edu/~kriz/cifar.html'
        },
        'cifar100': {
            'name': 'CIFAR-100',
            'size': '~170 MB',
            'samples': '60,000',
            'classes': 100,
            'url': 'https://www.cs.toronto.edu/~kriz/cifar.html'
        },
        'all_cifar': {
            'name': 'CIFAR-10 + CIFAR-100',
            'size': '~340 MB',
            'samples': '120,000',
            'classes': '10 + 100',
            'url': 'https://www.cs.toronto.edu/~kriz/cifar.html'
        }
    }
    
    if dataset_name in info:
        data = info[dataset_name]
        print(f"\n数据集信息:")
        print(f"  名称: {data['name']}")
        print(f"  大小: {data['size']}")
        print(f"  样本数: {data['samples']}")
        print(f"  类别数: {data['classes']}")
        print(f"  官网: {data['url']}")


def list_downloaded_datasets(data_path):
    """列出已下载的数据集"""
    print("="*60)
    print("检查已下载的数据集")
    print("="*60)
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"\n路径不存在: {data_path}")
        return
    
    print(f"\n检查路径: {data_path.absolute()}\n")
    
    # 检查 CIFAR-10
    cifar10_path = data_path / 'cifar-10-batches-py'
    if cifar10_path.exists():
        print("✓ CIFAR-10 已下载")
        print(f"  路径: {cifar10_path}")
        # 计算大小
        size = sum(f.stat().st_size for f in cifar10_path.rglob('*') if f.is_file())
        print(f"  大小: {size / (1024**2):.1f} MB")
    else:
        print("✗ CIFAR-10 未下载")
    
    # 检查 CIFAR-100
    cifar100_path = data_path / 'cifar-100-python'
    if cifar100_path.exists():
        print("\n✓ CIFAR-100 已下载")
        print(f"  路径: {cifar100_path}")
        size = sum(f.stat().st_size for f in cifar100_path.rglob('*') if f.is_file())
        print(f"  大小: {size / (1024**2):.1f} MB")
    else:
        print("\n✗ CIFAR-100 未下载")


def main():
    parser = argparse.ArgumentParser(
        description='下载和准备数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载 CIFAR-10 到默认路径
  python download_datasets.py --dataset cifar10
  
  # 下载 CIFAR-100 到指定路径
  python download_datasets.py --dataset cifar100 --path /data/cifar
  
  # 下载所有 CIFAR 数据集并验证
  python download_datasets.py --dataset all_cifar --verify
  
  # 列出已下载的数据集
  python download_datasets.py --list
  
  # 检查数据集信息
  python download_datasets.py --dataset cifar10 --info
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['cifar10', 'cifar100', 'all_cifar'],
        help='要下载的数据集'
    )
    
    parser.add_argument(
        '--path',
        type=str,
        default='./data/cifar',
        help='数据保存路径（默认: ./data/cifar）'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='下载后验证数据集'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出已下载的数据集'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='显示数据集信息'
    )
    
    parser.add_argument(
        '--check-space',
        action='store_true',
        help='检查磁盘空间'
    )
    
    args = parser.parse_args()
    
    # 列出已下载的数据集
    if args.list:
        list_downloaded_datasets(args.path)
        return
    
    # 显示数据集信息
    if args.info and args.dataset:
        show_download_info(args.dataset)
        return
    
    # 检查磁盘空间
    if args.check_space:
        check_disk_space(args.path)
        return
    
    # 下载数据集
    if args.dataset:
        # 检查磁盘空间
        check_disk_space(args.path)
        
        # 显示数据集信息
        show_download_info(args.dataset)
        
        print("\n" + "="*60)
        print("开始下载")
        print("="*60)
        
        if args.dataset == 'cifar10':
            download_cifar10(args.path, args.verify)
        elif args.dataset == 'cifar100':
            download_cifar100(args.path, args.verify)
        elif args.dataset == 'all_cifar':
            download_all_cifar(args.path, args.verify)
        
        print("\n" + "="*60)
        print("下载完成")
        print("="*60)
        print(f"\n使用方法:")
        print(f"  1. 在配置文件中设置路径:")
        print(f"     data:")
        print(f"       init_args:")
        print(f"         data_path: {args.path}")
        print(f"\n  2. 或在命令行指定路径:")
        print(f"     python main.py fit --config conf/config_cifar10.yaml \\")
        print(f"       --data.init_args.data_path={args.path}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

