"""
[已弃用] 兼容导入层

使用 motion_lib_cmg_realtime_v2.py 代替

旧版本问题：
- GPU高阶索引错误导致device-side asserts
- 缓冲机制不清晰
- 没有明确的同步推理策略

新版本（v2）特性：
✓ 每环境独立推理状态 + 2s参考缓冲
✓ 批量同步推理（避免异步问题）
✓ 清晰的推理触发条件
✓ 以当前关节角度为推理初始条件
"""

from .motion_lib_cmg_realtime_v2 import MotionLibCMGRealtime, create_motion_lib_cmg_realtime

__all__ = ['MotionLibCMGRealtime', 'create_motion_lib_cmg_realtime']
