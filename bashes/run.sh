export CUDA_VISIBLE_DEVICES=1
export SWANLAB_BASE_URL=http://154.64.255.168:8000
export SWANLAB_API_KEY=wHPCnBJpSPmxuZPVKULKi
# 验证配置（不实际训练）
# python main.py fit --config conf/config.yaml --print_config

# # 测试运行（快速验证）
# python main.py fit --config conf/config.yaml --trainer.fast_dev_run=true

# # 开始训练
python main.py fit --config conf/config_cifar10.yaml

# # 从 checkpoint 继续训练
# python main.py fit --config conf/config.yaml --ckpt_path checkpoints/last.ckpt

# # 仅验证
# python main.py validate --config conf/config.yaml --ckpt_path checkpoints/best.ckpt

# # 测试
# python main.py test --config conf/config.yaml --ckpt_path checkpoints/best.ckpt