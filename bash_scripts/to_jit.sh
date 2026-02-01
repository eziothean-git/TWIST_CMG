#!/bin/bash
# 导出JIT模型脚本
set -e

source ~/.bashrc
conda activate twist 2>/dev/null || echo "[WARN] conda环境twist未找到"

# 自动定位到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "[INFO] 工作目录: ${PROJECT_ROOT}"

# bash to_jit.sh 0927_twist_rlbcstu

exptid=${1}

proj_name="g1_stu_rl"

# Run the training script
python legged_gym/legged_gym/scripts/save_jit_stu_rlbc.py --robot "g1" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --checkpoint -1 \