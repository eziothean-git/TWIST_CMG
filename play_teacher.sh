#!/bin/bash
# Teacher模型测试脚本
# 
# 用法:
#   bash play_teacher.sh <exptid> [checkpoint] [num_envs] [record]
# 
# 参数:
#   exptid     - 实验ID (必需)
#   checkpoint - checkpoint编号, 默认 -1 (最新)
#   num_envs   - 环境数量, 默认 1
#   record     - 是否录制视频, 可选: record
#
# 示例:
#   bash play_teacher.sh test_cmg                    # 测试最新checkpoint
#   bash play_teacher.sh test_cmg 1200               # 测试第1200个checkpoint
#   bash play_teacher.sh test_cmg -1 4               # 4个环境
#   bash play_teacher.sh test_cmg -1 1 record        # 录制视频

set -e

source ~/.bashrc
conda activate twist 2>/dev/null || echo "[WARN] conda环境twist未找到"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

task_name="g1_priv_mimic"
proj_name="g1_priv_mimic"
exptid=$1
checkpoint=${2:--1}
num_envs=${3:-1}
record_mode=${4:-""}

if [ -z "$exptid" ]; then
    echo "错误: 必须提供实验ID"
    echo "用法: bash play_teacher.sh <exptid> [checkpoint] [num_envs] [record]"
    echo ""
    echo "示例:"
    echo "  bash play_teacher.sh test_cmg                    # 测试最新checkpoint"
    echo "  bash play_teacher.sh test_cmg 1200               # 测试第1200个checkpoint"
    echo "  bash play_teacher.sh test_cmg -1 1 record        # 录制视频"
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
echo "  Checkpoint: ${checkpoint}"
echo "  环境数:     ${num_envs}"
echo "=========================================="

cd "${SCRIPT_DIR}/legged_gym/legged_gym/scripts"

python3 play.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --checkpoint "${checkpoint}" \
                --num_envs "${num_envs}" \
                ${extra_args}