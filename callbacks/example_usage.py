"""
JiT Callbacks 使用示例

展示如何使用各种 callbacks 进行训练
"""

import argparse
import torch

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

# 假设这些模块已经存在
# from models.modelmodule import JiTLightningModule
# from datas.datamodule import JiTDataModule
from callbacks import (
    EMACallback,
    JiTModelCheckpoint,
    FIDEvaluationCallback,
    LearningRateSchedulerCallback,
    MetricLoggerCallback,
    create_default_callbacks,
)


# ============================================
# 方法 1: 使用单个 Callback
# ============================================
def example_single_callback():
    """使用单个 callback 的示例"""
    
    print("=" * 60)
    print("示例 1: 使用单个 Callback")
    print("=" * 60)
    
    # 创建 EMA Callback
    ema_callback = EMACallback(
        ema_decay1=0.9999,
        ema_decay2=0.9996,
    )
    
    # 创建模型检查点 Callback
    checkpoint_callback = JiTModelCheckpoint(
        dirpath='./checkpoints',
        save_last_freq=5,
        save_milestone_freq=100,
    )
    
    print(f"EMA Callback: {ema_callback}")
    print(f"Checkpoint Callback: {checkpoint_callback}")
    
    # 在 Trainer 中使用
    # trainer = pl.Trainer(
    #     callbacks=[ema_callback, checkpoint_callback],
    #     ...
    # )


# ============================================
# 方法 2: 使用所有默认 Callbacks
# ============================================
def example_default_callbacks():
    """使用所有默认 callbacks 的示例"""
    
    print("\n" + "=" * 60)
    print("示例 2: 使用所有默认 Callbacks")
    print("=" * 60)
    
    # 创建所有默认 callbacks
    callbacks = create_default_callbacks(
        ema_decay1=0.9999,
        ema_decay2=0.9996,
        save_dir='./checkpoints',
        save_last_freq=5,
        save_milestone_freq=100,
        eval_freq=40,
        num_images=50000,
        eval_batch_size=256,
        num_classes=1000,
        img_size=256,
        lr_schedule='constant',
        warmup_epochs=5,
        min_lr=0.0,
        epochs=600,
        log_freq=100,
        enable_fid_eval=True,
    )
    
    print(f"创建了 {len(callbacks)} 个 callbacks:")
    for i, cb in enumerate(callbacks, 1):
        print(f"  {i}. {cb.__class__.__name__}")
    
    # 在 Trainer 中使用
    # trainer = pl.Trainer(
    #     callbacks=callbacks,
    #     ...
    # )


# ============================================
# 方法 3: 完整的训练示例
# ============================================
def example_full_training():
    """完整的训练示例（需要实际的模型和数据）"""
    
    print("\n" + "=" * 60)
    print("示例 3: 完整的训练流程")
    print("=" * 60)
    
    # 创建 callbacks
    callbacks = create_default_callbacks(
        ema_decay1=0.9999,
        ema_decay2=0.9996,
        save_dir='./outputs/jit_b16_256',
        save_last_freq=5,
        save_milestone_freq=100,
        eval_freq=40,
        num_images=50000,
        eval_batch_size=256,
        num_classes=1000,
        img_size=256,
        lr_schedule='constant',
        warmup_epochs=5,
        epochs=600,
        enable_fid_eval=True,
    )
    
    # 创建 logger
    logger = TensorBoardLogger(
        save_dir='./logs',
        name='jit_experiment',
    )
    
    # 创建 Trainer
    trainer = pl.Trainer(
        max_epochs=600,
        accelerator='gpu',
        devices=8,
        strategy='ddp',
        precision='bf16-mixed',
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=100,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        enable_checkpointing=False,  # 使用自定义 checkpoint callback
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    print("\nTrainer 配置:")
    print(f"  - Max epochs: {trainer.max_epochs}")
    print(f"  - Devices: {trainer.num_devices}")
    print(f"  - Strategy: {trainer.strategy.__class__.__name__}")
    print(f"  - Callbacks: {len(trainer.callbacks)}")
    
    # 创建模型和数据模块（需要实际实现）
    # model = JiTLightningModule(...)
    # datamodule = JiTDataModule(...)
    
    # 开始训练
    # trainer.fit(model, datamodule=datamodule)


# ============================================
# 方法 4: 从 args 创建 Callbacks
# ============================================
def example_from_args():
    """从参数对象创建 callbacks"""
    
    print("\n" + "=" * 60)
    print("示例 4: 从 Args 创建 Callbacks")
    print("=" * 60)
    
    # 模拟 args 对象
    parser = argparse.ArgumentParser()
    
    # EMA 参数
    parser.add_argument('--ema_decay1', default=0.9999, type=float)
    parser.add_argument('--ema_decay2', default=0.9996, type=float)
    
    # 保存参数
    parser.add_argument('--output_dir', default='./checkpoints', type=str)
    parser.add_argument('--save_last_freq', default=5, type=int)
    
    # 评估参数
    parser.add_argument('--eval_freq', default=40, type=int)
    parser.add_argument('--num_images', default=50000, type=int)
    parser.add_argument('--gen_bsz', default=256, type=int)
    parser.add_argument('--class_num', default=1000, type=int)
    parser.add_argument('--img_size', default=256, type=int)
    
    # 学习率参数
    parser.add_argument('--lr_schedule', default='constant', type=str)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--min_lr', default=0.0, type=float)
    parser.add_argument('--epochs', default=600, type=int)
    
    # 日志参数
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--online_eval', action='store_true', default=True)
    
    args = parser.parse_args([])
    
    # 从 args 创建 callbacks
    callbacks = create_default_callbacks(
        ema_decay1=args.ema_decay1,
        ema_decay2=args.ema_decay2,
        save_dir=args.output_dir,
        save_last_freq=args.save_last_freq,
        save_milestone_freq=100,
        eval_freq=args.eval_freq,
        num_images=args.num_images,
        eval_batch_size=args.gen_bsz,
        num_classes=args.class_num,
        img_size=args.img_size,
        lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        epochs=args.epochs,
        log_freq=args.log_freq,
        enable_fid_eval=args.online_eval,
    )
    
    print(f"从 args 创建了 {len(callbacks)} 个 callbacks")


# ============================================
# 方法 5: 自定义 Callback 组合
# ============================================
def example_custom_combination():
    """自定义 callback 组合"""
    
    print("\n" + "=" * 60)
    print("示例 5: 自定义 Callback 组合")
    print("=" * 60)
    
    # 只使用部分 callbacks
    callbacks = []
    
    # 必须：EMA
    callbacks.append(EMACallback(ema_decay1=0.9999, ema_decay2=0.9996))
    
    # 必须：模型保存
    callbacks.append(JiTModelCheckpoint(
        dirpath='./checkpoints',
        save_last_freq=5,
    ))
    
    # 可选：学习率调度
    callbacks.append(LearningRateSchedulerCallback(
        lr_schedule='constant',
        warmup_epochs=5,
        epochs=600,
    ))
    
    # 可选：指标记录
    callbacks.append(MetricLoggerCallback(log_freq=100))
    
    # 注意：不包含 FID 评估（如果不需要）
    
    print(f"自定义组合包含 {len(callbacks)} 个 callbacks:")
    for cb in callbacks:
        print(f"  - {cb.__class__.__name__}")


# ============================================
# 方法 6: Callback 的高级用法
# ============================================
def example_advanced_usage():
    """Callback 的高级用法"""
    
    print("\n" + "=" * 60)
    print("示例 6: Callback 高级用法")
    print("=" * 60)
    
    # 创建 EMA callback
    ema_callback = EMACallback(ema_decay1=0.9999, ema_decay2=0.9996)
    
    # 模拟获取 EMA 状态字典
    print("\n高级功能：")
    print("1. 获取 EMA 状态字典")
    print("   ema_state_dict = ema_callback.get_ema_state_dict(model, ema_version=1)")
    
    print("\n2. 将 EMA 参数加载到模型")
    print("   ema_callback.load_ema_to_model(model, ema_version=1)")
    
    print("\n3. 在评估时使用 EMA 参数")
    print("""
    # 保存原始参数
    original_state = model.state_dict()
    
    # 切换到 EMA 参数
    ema_callback.load_ema_to_model(model, ema_version=1)
    
    # 评估
    evaluate(model)
    
    # 恢复原始参数
    model.load_state_dict(original_state)
    """)


# ============================================
# 方法 7: 与原始代码的对比
# ============================================
def example_comparison_with_original():
    """与原始代码的对比"""
    
    print("\n" + "=" * 60)
    print("示例 7: 与原始代码的对比")
    print("=" * 60)
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║              原始代码 vs Lightning Callbacks                ║
╚══════════════════════════════════════════════════════════════╝

原始代码（main_jit.py + engine_jit.py）:
─────────────────────────────────────────────────

for epoch in range(start_epoch, epochs):
    # 训练
    train_one_epoch(model, data_loader, optimizer, ...)
    
    # 手动保存 checkpoint
    if epoch % save_last_freq == 0:
        misc.save_model(args, model, optimizer, epoch, "last")
    
    if epoch % 100 == 0:
        misc.save_model(args, model, optimizer, epoch)
    
    # 手动评估
    if online_eval and epoch % eval_freq == 0:
        evaluate(model, args, epoch, ...)
    
    # 手动刷新日志
    if log_writer is not None:
        log_writer.flush()


Lightning Callbacks 方式:
─────────────────────────────────────────────────

# 创建 callbacks
callbacks = create_default_callbacks(
    save_last_freq=5,
    eval_freq=40,
    ...
)

# 创建 Trainer
trainer = pl.Trainer(
    max_epochs=600,
    callbacks=callbacks,
    ...
)

# 一行代码完成所有
trainer.fit(model, datamodule)

# 所有功能自动执行：
# ✓ EMA 更新
# ✓ Checkpoint 保存
# ✓ FID/IS 评估
# ✓ 学习率调度
# ✓ 日志记录


主要优势:
─────────────────────────────────────────────────
✓ 代码更简洁（从 ~100 行减少到 ~10 行）
✓ 自动处理分布式训练
✓ 更容易维护和扩展
✓ 更好的错误处理
✓ 统一的接口
✓ 更容易测试
    """)


if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           JiT Lightning Callbacks 使用示例                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    example_single_callback()
    example_default_callbacks()
    example_full_training()
    example_from_args()
    example_custom_combination()
    example_advanced_usage()
    example_comparison_with_original()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)

