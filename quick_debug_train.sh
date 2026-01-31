#!/bin/bash
# 快速调试运行：仅运行50步来查看 reward 分解
cd /home/eziothean/TWIST_CMG
python train_teacher.sh 2>&1 | head -n 500 > /tmp/training_debug.log
# 显示调试信息
tail -n 200 /tmp/training_debug.log
