#!/bin/bash
# Teacher训练脚本
# 
# 用法:
#   bash train_teacher.sh <exptid> [device] [num_envs] [mode] [extra]
# 
# 参数:
#   exptid   - 实验ID (必需)
#   device   - GPU设备, 默认 cuda:0
#   num_envs - 环境数量, 默认不指定(使用config值), 可选: 32, 64, 128, 256, 512, 1024, 2048, 4096
#   mode     - 运行模式, 默认normal, 可选: debug(32 envs + 可视化)
#   extra    - 额外选项, 可选: bg(后台运行), resume(继续训练), bg+resume(两者都要)
#
# 示例:
#   bash train_teacher.sh test_cmg                            # 新训练
#   bash train_teacher.sh test_cmg cuda:0 256                 # 256个环境
#   bash train_teacher.sh test_cmg cuda:0 0 normal resume     # 继续训练
#   bash train_teacher.sh test_cmg cuda:0 0 normal bg         # 后台运行
#   bash train_teacher.sh test_cmg cuda:0 256 normal bg+resume # 后台继续训练

set -e

# 激活conda环境
source ~/.bashrc
conda activate twist 2>/dev/null || echo "[WARN] conda环境twist未找到，使用当前环境"

# 切换到脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exptid=$1
device=${2:-cuda:0}
num_envs=${3:-0}
mode=${4:-normal}
extra=${5:-""}

task_name="g1_priv_mimic"
proj_name="g1_priv_mimic"
LOG_DIR="${SCRIPT_DIR}/logs"

if [ -z "$exptid" ]; then
    echo "错误: 必须提供实验ID"
    echo "用法: bash train_teacher.sh <exptid> [device] [num_envs] [mode] [extra]"
    echo ""
    echo "示例:"
    echo "  bash train_teacher.sh test_cmg                            # 新训练"
    echo "  bash train_teacher.sh test_cmg cuda:0 0 normal resume     # 继续训练"
    echo "  bash train_teacher.sh test_cmg cuda:0 0 normal bg         # 后台运行"
    echo "  bash train_teacher.sh test_cmg cuda:0 256 normal bg+resume # 后台继续训练"
    exit 1
fi

# 构建命令参数
extra_args=""

# 处理运行模式
if [ "$mode" = "debug" ]; then
    extra_args="${extra_args} --debug"
    echo "[INFO] Debug模式: 32环境 + 可视化 + 禁用wandb"
fi

# 处理环境数量覆盖
if [ "$num_envs" -gt 0 ] 2>/dev/null; then
    extra_args="${extra_args} --num_envs ${num_envs}"
    echo "[INFO] 覆盖环境数量: ${num_envs}"
fi

# 检查是否需要resume
bg_mode=""
if [[ "$extra" == *"resume"* ]]; then
    extra_args="${extra_args} --resume"
    echo "[INFO] 继续训练模式"
fi
if [[ "$extra" == *"bg"* ]]; then
    bg_mode="bg"
fi

echo "=========================================="
echo "  TWIST Teacher 训练"
echo "=========================================="
echo "  实验ID:   ${exptid}"
echo "  设备:     ${device}"
echo "  任务:     ${task_name}"
echo "  模式:     ${mode}"
if [ "$num_envs" -gt 0 ] 2>/dev/null; then
    echo "  环境数:   ${num_envs}"
fi
if [[ "$extra" == *"resume"* ]]; then
    echo "  继续训练: 是"
fi
if [ "$bg_mode" = "bg" ]; then
    echo "  后台运行: 是"
fi
echo "=========================================="

cd "${SCRIPT_DIR}/legged_gym/legged_gym/scripts"

# 构建完整命令
CMD="python3 train.py --task ${task_name} --proj_name ${proj_name} --exptid ${exptid} --device ${device} ${extra_args}"

# 后台运行模式
if [ "$bg_mode" = "bg" ]; then
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/train_${exptid}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "[INFO] 后台运行模式，日志保存到: ${LOG_FILE}"
    echo "[INFO] 查看日志: tail -f ${LOG_FILE}"
    echo "[INFO] 查看进程: ps aux | grep train.py"
    echo "[INFO] 停止训练: pkill -f \"exptid ${exptid}\""
    
    nohup $CMD > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "[INFO] 训练已在后台启动，PID: ${PID}"
    
    # 等待几秒确认启动成功
    sleep 3
    if ps -p $PID > /dev/null 2>&1; then
        echo "[SUCCESS] 训练进程正在运行"
        echo ""
        echo "最新日志:"
        tail -n 10 "${LOG_FILE}"
    else
        echo "[ERROR] 训练进程启动失败，查看日志: cat ${LOG_FILE}"
        exit 1
    fi
else
    # 前台运行
    $CMD
fi
