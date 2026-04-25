#!/bin/bash
# ==============================================
# Pointcept 断点续训专用脚本
# 用法: ./resume_train.sh <实验目录路径>
# 示例: ./resume_train.sh exp/s3dis/debug_20260120_1430
# ==============================================

# --- 基本路径设置 ---
cd "$(dirname "$0")" || exit
ROOT_DIR=$(pwd)/..

# --- 默认参数 ---
PYTHON=python
NUM_GPU=None
NUM_MACHINE=1
DIST_URL="auto"

# --- 解析命令行参数 ---
EXP_DIR=""
while getopts "e:p:g:" opt; do
  case $opt in
    e)
      EXP_DIR=$OPTARG
      ;;
    p)
      PYTHON=$OPTARG
      ;;
    g)
      NUM_GPU=$OPTARG
      ;;
    \?)
      echo "无效选项: -$OPTARG"
      exit 1
      ;;
  esac
done

# 如果第一个非选项参数存在，也视为 EXP_DIR (兼容位置参数)
if [ -z "$EXP_DIR" ] && [ -n "$1" ]; then
  EXP_DIR="$1"
fi

# --- 检查实验目录是否有效 ---
if [ -z "$EXP_DIR" ]; then
  echo "错误: 请指定实验目录路径"
  echo "用法: $0 [-e] <实验目录>"
  echo "示例: $0 exp/s3dis/debug_20260120_1430"
  exit 1
fi

if [ ! -d "$EXP_DIR" ]; then
  echo "错误: 目录不存在: $EXP_DIR"
  exit 1
fi

# --- 自动推断必要路径 ---
CONFIG_DIR="${EXP_DIR}/config.py"
MODEL_DIR="${EXP_DIR}/model"
WEIGHT="${MODEL_DIR}/model_last.pth"
CODE_DIR="${EXP_DIR}/code"

# --- 检查必要文件是否存在 ---
if [ ! -f "$CONFIG_DIR" ]; then
  echo "错误: 配置文件不存在: $CONFIG_DIR"
  exit 1
fi

if [ ! -f "$WEIGHT" ]; then
  echo "错误: 检查点文件不存在: $WEIGHT"
  exit 1
fi

# 强制使用当前源码目录（包含修复）
echo "注意: 使用当前源码目录运行训练"
CODE_DIR="$ROOT_DIR"

# --- 设置 Python 搜索路径 ---
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

# --- 自动检测 GPU 数量 ---
if [ "${NUM_GPU}" = 'None' ]; then
  NUM_GPU=$($PYTHON -c 'import torch; print(torch.cuda.device_count())')
fi

# --- 显示恢复信息 ---
echo "=========================================="
echo "   Pointcept 断点续训"
echo "=========================================="
echo "实验目录:     $EXP_DIR"
echo "配置文件:     $CONFIG_DIR"
echo "权重文件:     $WEIGHT"
echo "代码目录:     $CODE_DIR"
echo "GPU 数量:     $NUM_GPU"
echo "Python 解释器: $PYTHON"
echo "=========================================="

# --- 执行续训命令 ---
ulimit -n 65536

$PYTHON "$CODE_DIR"/tools/train.py \
  --config-file "$CONFIG_DIR" \
  --num-gpus "$NUM_GPU" \
  --num-machines "$NUM_MACHINE" \
  --machine-rank 0 \
  --dist-url "$DIST_URL" \
  --options save_path="$EXP_DIR" resume=true weight="$WEIGHT"

# --- 训练完成后自动关闭服务器 ---
echo "=========================================="
echo "训练已完成，5分钟后自动关闭服务器..."
echo "=========================================="
sleep 300
shutdown -h now
