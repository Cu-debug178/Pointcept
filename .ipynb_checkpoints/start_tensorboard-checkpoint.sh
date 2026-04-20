#!/bin/bash
# ============================================
# TensorBoard 管理脚本 (AutoDL 环境)
# 功能：
#   1. 关闭所有正在运行的 TensorBoard 进程
#   2. 启动一个新的 TensorBoard 服务，指向指定的日志根目录
# 使用方法：
#   bash start_tensorboard.sh
# ============================================

# ========== ！！！需要根据实际情况修改的参数 ==========

# ！ 需要修改：日志根目录（TensorBoard 会递归扫描该目录下的所有子目录）
# 说明：请改为你实际存放多个实验日志的父目录路径
LOGDIR="/root/autodl-tmp/Pointcept/exp/"

# ！ 需要修改：TensorBoard 服务端口（必须与 SSH 隧道转发的远程端口一致）
# 说明：通常用 6006，如果被占用可改为 6007、6008 等
PORT=6006

# ！ 通常不需要修改：绑定的主机地址（必须为 0.0.0.0，否则 SSH 隧道无法转发）
HOST="0.0.0.0"

# ！ 可选修改：nohup 输出的日志文件保存路径
# 说明：脚本运行后会在当前目录生成 tensorboard.log，你可以改成绝对路径
LOG_FILE="tensorboard.log"

# =================================================

# ---------- 1. 关闭现有的 TensorBoard 进程 ----------
echo "[1/3] 正在查找并关闭现有 TensorBoard 进程..."

# 方法1：使用 pkill（更安全，仅杀死 tensorboard 进程）
if command -v pkill &> /dev/null; then
    pkill -f "tensorboard" && echo "已终止 tensorboard 进程" || echo "没有找到运行中的 tensorboard 进程"
else
    # 方法2：使用 ps + grep + kill（兼容性较好）
    TB_PIDS=$(ps -ef | grep "[t]ensorboard" | awk '{print $2}')
    if [ -n "$TB_PIDS" ]; then
        echo "找到 TensorBoard 进程 PID: $TB_PIDS"
        kill -9 $TB_PIDS
        echo "已强制终止上述进程"
    else
        echo "没有找到运行中的 tensorboard 进程"
    fi
fi

# 可选：等待一秒确保端口释放
sleep 1

# ---------- 2. 检查日志目录是否存在 ----------
echo "[2/3] 检查日志目录: $LOGDIR"
if [ ! -d "$LOGDIR" ]; then
    echo "警告: 日志目录不存在，正在创建..."
    mkdir -p "$LOGDIR"
    if [ $? -eq 0 ]; then
        echo "目录创建成功: $LOGDIR"
    else
        echo "错误: 无法创建目录，请检查路径权限"
        exit 1
    fi
else
    echo "日志目录已存在"
fi

# ---------- 3. 启动新的 TensorBoard 服务 ----------
echo "[3/3] 正在启动 TensorBoard..."
echo "  日志根目录: $LOGDIR"
echo "  监听端口: $PORT"
echo "  监听地址: $HOST"

# 启动 TensorBoard（后台运行，并将输出重定向到日志文件，避免占用终端）
nohup tensorboard --logdir="$LOGDIR" --port="$PORT" --host="$HOST" > "$LOG_FILE" 2>&1 &

# 获取新启动的进程 PID
NEW_PID=$!
echo "TensorBoard 已启动，进程 PID: $NEW_PID"
echo "日志输出文件: $(pwd)/$LOG_FILE"

# 等待 2 秒检查服务是否真正启动
sleep 2
if ps -p $NEW_PID > /dev/null; then
    echo "✅ TensorBoard 启动成功！"
    echo ""
    echo "========== 下一步：在本地电脑执行 SSH 隧道命令 =========="
    echo "！ 需要修改：请将下面命令中的 <本地端口>、<服务器地址>、<端口号> 替换为你的实际值"
    echo "  ssh -L <本地端口>:127.0.0.1:$PORT root@<服务器地址> -p <端口号>"
    echo "然后在本地浏览器访问: http://localhost:<本地端口>"
    echo "======================================================"
else
    echo "❌ TensorBoard 启动失败，请检查错误日志: tail -f $LOG_FILE"
    exit 1
fi