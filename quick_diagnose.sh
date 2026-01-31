#!/bin/bash
# 快速诊断脚本：仅运行50步训练并收集调试信息

set -e

cd /home/eziothean/TWIST_CMG

echo "=========================================="
echo "CMG 训练快速诊断 (50步)"
echo "=========================================="
echo ""
echo "记录位置: /tmp/cmg_diagnosis.log"
echo ""

# 运行50步训练，保存日志
python -m legged_gym.envs.adapt_env \
  with model=teacher \
  num_envs=64 \
  max_iterations=50 \
  2>&1 | tee /tmp/cmg_diagnosis.log

echo ""
echo "=========================================="
echo "诊断结果"
echo "=========================================="
echo ""

# 提取关键信息
echo "【RESET 时间信息】"
grep "DEBUG RESET" /tmp/cmg_diagnosis.log | head -2 || echo "（未找到RESET调试信息）"

echo ""
echo "【Step 1 参考检查】"
grep "DEBUG Step 1" /tmp/cmg_diagnosis.log | head -20 || echo "（未找到Step 1调试信息）"

echo ""
echo "【Reward 分解 (Step 1)】"
grep -A 15 "Reward 分解" /tmp/cmg_diagnosis.log | head -20 || echo "（未找到Reward信息）"

echo ""
echo "=========================================="
echo "诊断完成！"
echo "完整日志: /tmp/cmg_diagnosis.log"
echo "关键指标："
echo "  - 查看 RESET 时的 motion_time 值"
echo "  - 查看 Step 1 时的 motion_time 值（应该相同）"
echo "  - 查看 tracking_joint_dof reward 值（应该 > 0.2）"
echo "=========================================="
