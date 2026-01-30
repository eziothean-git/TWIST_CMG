#!/bin/bash
# Teacher训练脚本
# 
# 用法:
#   bash train_teacher.sh <exptid> [device] [num_envs] [mode]
# 
# 参数:
#   exptid   - 实验ID (必需)
#   device   - GPU设备, 默认 cuda:0
#   num_envs - 环境数量, 默认不指定(使用config值), 可选: 32, 64, 128, 256, 512, 1024, 2048, 4096
#   mode     - 运行模式, 默认normal, 可选: debug(32 envs + 可视化), headless
#
# 示例:
#   bash train_teacher.sh test_cmg                      # 默认配置
#   bash train_teacher.sh test_cmg cuda:1               # 指定GPU
#   bash train_teacher.sh test_cmg cuda:0 256           # 256个环境
#   bash train_teacher.sh test_cmg cuda:0 32 debug      # debug模式 (32 envs + 可视化)
#   bash train_teacher.sh test_cmg cuda:0 0 debug       # debug模式 (使用debug默认的32 envs)

set -e

# 激活conda环境
source ~/.bashrc
conda activate twist 2>/dev/null || echo "[WARN] conda环境twist未找到，使用当前环境"

# 切换到脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/legged_gym/legged_gym/scripts"

exptid=$1
device=${2:-cuda:0}
num_envs=${3:-0}
mode=${4:-normal}

task_name="g1_priv_mimic"
proj_name="g1_priv_mimic"

if [ -z "$exptid" ]; then
    echo "错误: 必须提供实验ID"
    echo "用法: bash train_teacher.sh <exptid> [device] [num_envs] [mode]"
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
echo "=========================================="

# Run the training script
python3 train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                ${extra_args}
                # --resume \
                # --resumeid xxx
