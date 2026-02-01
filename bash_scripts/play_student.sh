#!/bin/bash
# Student模型测试脚本
set -e

source ~/.bashrc
conda activate twist 2>/dev/null || echo "[WARN] conda环境twist未找到"

# 自动定位到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "[INFO] 工作目录: ${PROJECT_ROOT}"

# bash play_student.sh g1 0927_twist_rlbcstu 0927_twist_teacher

exptid=$1
teacher_exptid=$2
task_name="g1_stu_rl"
proj_name="g1_stu_rl"

# Run the eval script
python legged_gym/legged_gym/scripts/play.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --num_envs 1 \
                --teacher_exptid "${teacher_exptid}" \
                --record_video \
