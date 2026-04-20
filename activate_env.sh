#!/bin/bash

# Pointcept虚拟环境激活脚本
# 使用方法: source activate_env.sh 或 . activate_env.sh

# 激活conda
source /root/miniconda3/etc/profile.d/conda.sh

# 激活pointcept虚拟环境
conda activate /root/autodl-tmp/envs/pointcept

# 设置工作目录
cd /root/autodl-tmp/Pointcept

# 设置PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/Pointcept

# 设置CUDA内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "✅ Pointcept虚拟环境已激活"
echo "📁 当前目录: $(pwd)"
echo "🐍 Python版本: $(python --version 2>&1)"
echo "🚀 可以开始运行训练或测试命令了"