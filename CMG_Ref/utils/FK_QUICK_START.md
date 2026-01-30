# FK集成快速指南

## 30秒快速开始

### 启用CMG+FK集成

```python
from CMG_Ref.utils.cmg_motion_generator import CMGMotionGenerator

gen = CMGMotionGenerator(
    model_path="model.pt",
    data_path="data.pt", 
    fk_model_path="assets/g1/g1_29dof.urdf",  # ← 新增
    enable_fk=True,  # ← 新增
)

result = gen.get_motion_with_body_transforms()
# result包含: dof_pos, dof_vel, body_pos, body_rot
```

## 核心功能

### 1. 关节到body位置/旋转

```python
from CMG_Ref.utils.fk_integration import compute_body_transforms_from_dof
from pose.util_funcs.kinematics_model import KinematicsModel

fk = KinematicsModel("assets/g1/g1_29dof.urdf", "cuda")

result = compute_body_transforms_from_dof(
    dof_positions,        # [batch, 29]
    dof_velocities,       # [batch, 29]
    fk_model=fk,
    base_pos=base_pos,    # [batch, 3]
    base_rot=base_rot,    # [batch, 4] 四元数wxyz
)

body_pos = result['body_positions']   # [batch, 13, 3]
body_rot = result['body_rotations']   # [batch, 13, 4]
```

### 2. 获取default key bodies

```python
from CMG_Ref.utils.fk_integration import get_default_key_bodies

bodies = get_default_key_bodies()
# ['pelvis', 'left_hip_pitch_link', ..., 'left_wrist_roll_link', 
#  'right_wrist_roll_link']
```

### 3. 验证FK实现

```python
from CMG_Ref.utils.fk_integration import validate_fk_implementation
from pose.util_funcs.kinematics_model import KinematicsModel

fk = KinematicsModel("assets/g1/g1_29dof.urdf", "cuda")
is_valid = validate_fk_implementation(fk)
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `pose/pose/util_funcs/kinematics_model.py` | 核心FK实现 (29 DOF支持) |
| `CMG_Ref/utils/fk_integration.py` | FK工具和辅助函数 |
| `CMG_Ref/utils/cmg_motion_generator.py` | CMG+FK集成 |
| `CMG_Ref/FK_IMPLEMENTATION_README.md` | 详细文档 |
| `CMG_Ref/example_fk_integration.py` | 使用示例 |

## 关键参数

### KinematicsModel
- `file_path`: URDF或XML文件路径
- `device`: 'cuda' 或 'cpu'

### forward_kinematics()
- `joint_angles`: [batch, 29] DOF位置
- `base_pos`: [batch, 3] 基座位置
- `base_rot`: [batch, 4] 基座旋转 (wxyz)
- `key_bodies`: body名称列表

### 返回值
- `body_positions`: [batch, num_bodies, 3]
- `body_rotations`: [batch, num_bodies, 4]

## 输出格式

```python
# CMG生成器返回
result = {
    'dof_positions': [batch, 29],           # 关节位置
    'dof_velocities': [batch, 29],          # 关节速度
    'body_positions': [batch, 13, 3],       # 13个body的全局位置
    'body_rotations': [batch, 13, 4],       # 四元数 wxyz
}
```

## 29 DOF body列表

1. pelvis
2. left_hip_pitch_link
3. left_knee_link
4. left_ankle_pitch_link
5. right_hip_pitch_link
6. right_knee_link
7. right_ankle_pitch_link
8. left_shoulder_pitch_link
9. left_elbow_link
10. **left_wrist_roll_link** ✨ (新增)
11. right_shoulder_pitch_link
12. right_elbow_link
13. **right_wrist_roll_link** ✨ (新增)

## 常见用途

### 在TWIST中计算奖励
```python
# 计算body位置跟踪误差
ref_body_pos, ref_body_rot = fk.forward_kinematics(
    ref_dof, base_pos, base_rot, key_bodies
)
actual_body_pos = obs['body_pos']

reward_pos = -torch.norm(actual_body_pos - ref_body_pos, dim=-1).mean()
```

### 离线转换NPZ到PKL
```python
from CMG_Ref.utils.fk_integration import npz_to_pkl_with_fk

fk = KinematicsModel("assets/g1/g1_29dof.urdf", "cuda")
npz_data = np.load("motion.npz")
pkl_data = npz_to_pkl_with_fk(npz_data, fk_model=fk)

with open("motion_with_fk.pkl", "wb") as f:
    pickle.dump(pkl_data, f)
```

## 故障排除

### 问题: FK模型加载失败
**解决**: 检查URDF/XML文件路径
```python
from pathlib import Path
assert Path("assets/g1/g1_29dof.urdf").exists()
```

### 问题: 关节数量不匹配
**解决**: FK模型会自动检测和适配，无需手动干预

### 问题: 性能不佳
**解决**: 使用CUDA设备
```python
fk = KinematicsModel(path, device='cuda')  # 而非 'cpu'
```

## 详细文档

完整文档和更多示例见 `CMG_Ref/FK_IMPLEMENTATION_README.md`

## 相关任务

- ✅ 1.1.3 前向运动学实现
- → 1.1.2 运动格式转换 (已支持FK计算)
- → 2.1.1 CMG-TWIST桥接类 (已集成FK)
