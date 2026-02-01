#!/bin/bash
# Teacher模型测试脚本
# 
# 用法:
#   bash play_teacher.sh <exptid> [device] [checkpoint] [num_envs] [record] [task_type]
# 
# 参数:
#   exptid     - 实验ID (必需)
#   device     - GPU设备, 默认 cuda:0
#   checkpoint - checkpoint编号, 默认 -1 (最新)
#   num_envs   - 环境数量, 默认 1
#   record     - 是否录制视频, 可选: record
#   task_type  - 任务类型, 默认priv, 可选: priv, cmg_slow, cmg_medium, cmg_fast
#
# 示例:
#   bash play_teacher.sh test_cmg                             # 默认priv测试最新checkpoint
#   bash play_teacher.sh test_cmg cuda:0 1200                 # 测试第1200个checkpoint
#   bash play_teacher.sh test_cmg cuda:0 -1 4                 # 4个环境
#   bash play_teacher.sh test_cmg cuda:0 -1 1 record          # 录制视频
#   bash play_teacher.sh test_cmg cuda:0 -1 1 "" cmg_medium  # CMG中速任务

set -e

source ~/.bashrc
conda activate twist 2>/dev/null || echo "[WARN] conda环境twist未找到"

# 自动定位到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "[INFO] 工作目录: ${PROJECT_ROOT}"

exptid=$1
device=${2:-cuda:0}
checkpoint=${3:--1}
num_envs=${4:-1}
record_mode=${5:-""}
task_type=${6:-priv}

# 根据 task_type 选择任务名和项目名
case $task_type in
    priv)
        task_name="g1_priv_mimic"
        proj_name="g1_priv_mimic"
        ;;
    cmg_slow)
        task_name="g1_cmg_slow"
        proj_name="g1_cmg_slow"
        ;;
    cmg_medium)
        task_name="g1_cmg_medium"
        proj_name="g1_cmg_medium"
        ;;
    cmg_fast)
        task_name="g1_cmg_fast"
        proj_name="g1_cmg_fast"
        ;;
    *)
        echo "错误: 无效的task_type: $task_type"
        echo "可选: priv, cmg_slow, cmg_medium, cmg_fast"
        exit 1
        ;;
esac

if [ -z "$exptid" ]; then
    echo "错误: 必须提供实验ID"
    echo "用法: bash play_teacher.sh <exptid> [device] [checkpoint] [num_envs] [record] [task_type]"
    echo ""
    echo "示例:"
    echo "  bash play_teacher.sh test_cmg                             # 默认priv测试最新checkpoint"
    echo "  bash play_teacher.sh test_cmg cuda:0 1200                 # 测试第1200个checkpoint"
    echo "  bash play_teacher.sh test_cmg cuda:0 -1 1 record          # 录制视频"
    echo "  bash play_teacher.sh test_cmg cuda:0 -1 1 \"\" cmg_medium  # CMG中速任务"
    exit 1
fi

extra_args=""
if [ "$record_mode" = "record" ]; then
    extra_args="${extra_args} --record_video"
    echo "[INFO] 录制视频模式"
fi

echo "=========================================="
echo "  TWIST Teacher 测试"
echo "=========================================="
echo "  实验ID:     ${exptid}"
echo "  设备:       ${device}"
echo "  任务:       ${task_name}"
echo "  Checkpoint: ${checkpoint}"
echo "  环境数:     ${num_envs}"
echo "=========================================="

python3 legged_gym/legged_gym/scripts/play.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --checkpoint "${checkpoint}" \
                --num_envs "${num_envs}" \
                ${extra_args}