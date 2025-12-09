"""
JiT 训练主程序
使用 LightningCLI 从配置文件加载参数

用法:
    # 使用默认配置训练
    python main.py fit --config conf/config.yaml
    
    # 使用其他配置
    python main.py fit --config conf/config_jit_l16_256.yaml
    
    # 命令行覆盖配置
    python main.py fit --config conf/config.yaml --trainer.max_epochs=300
    
    # 从 checkpoint 恢复训练
    python main.py fit --config conf/config.yaml --ckpt_path path/to/checkpoint.pth
    
    # 查看完整配置
    python main.py fit --config conf/config.yaml --print_config
"""

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import LightningModule, LightningDataModule
import warnings

# 防止警告信息
warnings.filterwarnings("ignore", message=".*Only one live display.*")


class JiTLightningCLI(LightningCLI):
    """
    JiT 自定义 LightningCLI
    
    支持从 YAML 配置文件加载所有参数，包括：
    - Model (JiTLightningModule)
    - Data (JiTDataModule)
    - Callbacks (EMA, Checkpoint, FID Evaluation, etc.)
    - Trainer (分布式、精度、优化等)
    """
    
    def add_arguments_to_parser(self, parser):
        """
        添加自定义参数链接
        
        可以在这里链接不同模块之间的参数，确保一致性
        """
        # 确保 model 和 data 的 img_size 一致
        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size", apply_on="instantiate")
        
        # 可以添加更多参数链接
        # parser.link_arguments("data.init_args.batch_size", "model.init_args.batch_size")
    
    def before_instantiate_classes(self) -> None:
        """在实例化类之前调用，用于设置默认 logger"""
        super().before_instantiate_classes()
        
        # 如果配置文件中没有指定 logger，使用默认的 TensorBoard logger
        if hasattr(self.config, 'fit') and not hasattr(self.config.fit.trainer, 'logger'):
            from lightning.pytorch.loggers import TensorBoardLogger
            # 默认 logger 会自动创建，无需手动设置
            pass

cli = JiTLightningCLI(
    LightningModule,
    LightningDataModule,
    subclass_mode_model=True,   # 允许从配置文件指定 model class_path
    subclass_mode_data=True,    # 允许从配置文件指定 data class_path
    save_config_callback=None,  # 禁用自动保存配置（使用自定义 checkpoint）
)

