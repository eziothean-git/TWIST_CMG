# DOF对齐策略 - 修正版

## 关键理解

**架构关系：**
```
CMG (已训练好，29 DOF) 
   ↓ 输出参考动作
TWIST 残差网络 (正在训练) 
   ↓ 学习跟踪CMG + 残差修正
实机部署 (29 DOF)
```

**核心问题：**
- ✅ CMG 已训练完成，输出 29 DOF
- ❌ TWIST 当前使用 23 DOF URDF训练
- ✅ 实机是 29 DOF
- ❓ 应该谁对齐谁？

---

## 结论：TWIST应该使用29 DOF训练

### 理由

1. **CMG是固定的**
   - CMG已经训练好，模型frozen
   - CMG输出29 DOF运动数据作为参考
   - 不应该为了TWIST去修改CMG

2. **TWIST是残差学习**
   - TWIST的任务是学习：`final_action = CMG_reference + residual`
   - 输入包含CMG的参考动作（29维）
   - 如果TWIST只有23 DOF，维度就对不上

3. **端到端对齐**
   ```
   训练：CMG(29) → TWIST(29) → 仿真机器人(29)
   部署：CMG(29) → TWIST(29) → 实机(29)
   ```
   
4. **实机是29 DOF**
   - 最终部署目标有29个电机
   - 训练时就应该和实机一致

---

## 方案对比（修正版）

### ❌ 方案A：保持TWIST 23 DOF + 转换器

**流程：**
```
CMG输出(29) → [丢弃手腕6DOF] → TWIST训练(23) 
             → [部署时补充手腕] → 实机(29)
```

**问题：**
1. **训练和部署不一致**
   - 训练时没见过手腕DOF
   - 部署时手腕是固定值或规则
   - sim2real gap增大

2. **浪费CMG信息**
   - CMG生成的手腕运动被丢弃
   - TWIST学不到手腕协调

3. **残差学习困难**
   - CMG参考29维，TWIST只能输出23维
   - 观测中需要手动裁剪CMG参考
   - 不优雅且容易出错

**唯一优点：** 无需重新训练TWIST（但TWIST正在训练中！）

---

### ✅ 方案B：TWIST使用29 DOF URDF（推荐）

**流程：**
```
CMG输出(29) → TWIST训练(29) → 仿真(29) → 实机(29)
```

**优点：**
1. **完全对齐**
   - CMG、TWIST、实机都是29 DOF
   - 无需任何转换
   - sim2real直接迁移

2. **充分利用CMG**
   - TWIST能学习CMG的完整运动
   - 包括手腕的精细控制
   - 残差修正更准确

3. **架构简洁**
   ```python
   # 清晰的残差学习
   cmg_ref = cmg_model(command)           # (29,)
   obs = build_obs(..., cmg_ref)          # 包含29维参考
   residual = twist_policy(obs)           # (29,)
   final_action = cmg_ref + residual      # (29,)
   ```

4. **TWIST正在训练中**
   - **现在修改配置代价最小**
   - 比训练完再改简单得多
   - 一次性做对

**缺点：**
- 需要修改配置文件（但很简单）
- 需要重新开始训练（但本来就在训练）

---

## 实施方案B的步骤

### 复杂度评估：⭐⭐ (中等，但值得)

### 需要修改的文件

#### 1. 配置文件（核心修改）

```python
# legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py

class G1MimicPrivCfg(HumanoidMimicCfg):
    class env:
        num_actions = 29  # ← 23改为29
        
        # 自动重新计算依赖变量
        n_priv_latent = 4 + 1 + 2*num_actions
        n_proprio = 3 + 2 + 3*num_actions
        # 其他自动适配
    
    class asset:
        # 替换URDF
        file = f'{LEGGED_GYM_ROOT_DIR}/../assets/g1/g1_29dof_rev_1_0.urdf'
        
        # 更新armature（添加6个手腕）
        dof_armature = [
            0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597,  # 左腿(6)
            0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597,  # 右腿(6)
            0.0103, 0.0103, 0.0103,                               # 腰(3)
            0.003597, 0.003597, 0.003597, 0.003597,               # 左臂(4)
            0.00425, 0.00425, 0.00425,                            # 左手腕(3) ← 新增
            0.003597, 0.003597, 0.003597, 0.003597,               # 右臂(4)
            0.00425, 0.00425, 0.00425,                            # 右手腕(3) ← 新增
        ]
    
    class init_state:
        default_joint_angles = {
            # ... 原有23个关节 ...
            
            # 添加6个手腕关节（从CMG数据或合理初值）
            'left_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
        }
    
    class control:
        stiffness = {
            # ... 原有 ...
            'wrist': 20,  # 新增
        }
        damping = {
            # ... 原有 ...
            'wrist': 1,   # 新增
        }
```

#### 2. 检查硬编码（可能不需要改）

大部分代码已经使用 `self.num_dof`，会自动适配：
```python
# 这些会自动变成29
self.num_dof = self.gym.get_asset_dof_count(robot_asset)  # 从URDF读取
self.dof_pos = torch.zeros(self.num_envs, self.num_dof, ...)
```

**需要检查的地方：**
```bash
# 搜索硬编码的23
cd legged_gym/legged_gym/envs/g1/
grep -n "23" *.py | grep -v "2023\|#"
```

可能需要改：
- 奖励权重列表（如果有23个元素）
- 索引数组（如果硬编码了关节索引）

#### 3. Motion Library适配（关键）

**检查CMG数据的关节顺序：**
```python
# CMG_Ref/eval_cmg.py
dof_positions = motion[:, :29]  # 29个关节

# 需要确认：
# CMG的29个关节顺序 == 29dof_rev_1_0.urdf的关节顺序？
```

**如果顺序不同，需要映射：**
```python
# pose/pose/utils/motion_lib.py 或自定义加载器
def load_cmg_motion(cmg_npz):
    dof_pos = cmg_npz['dof_positions']  # (T, 29)
    
    # 如果需要重排序
    cmg_to_urdf_indices = [
        0,1,2,3,4,5,      # 左腿
        6,7,8,9,10,11,    # 右腿
        12,13,14,         # 腰
        15,16,17,18,      # 左臂
        19,20,21,         # 左手腕
        22,23,24,25,      # 右臂
        26,27,28,         # 右手腕
    ]
    dof_pos_reordered = dof_pos[:, cmg_to_urdf_indices]
    
    return dof_pos_reordered
```

---

## 具体实施清单

### ✅ 立即执行（修改配置）

**Step 1: 更新配置（10分钟）**
```python
# 修改 g1_mimic_distill_config.py
- num_actions = 23
+ num_actions = 29

- file = '.../g1_custom_collision_with_fixed_hand.urdf'
+ file = '.../g1_29dof_rev_1_0.urdf'

+ dof_armature 添加6个手腕值
+ default_joint_angles 添加6个手腕初值
+ stiffness/damping 添加wrist项
```

**Step 2: 检查硬编码（10分钟）**
```bash
grep -rn "23" legged_gym/legged_gym/envs/g1/*.py
# 修改任何硬编码的23为29或self.num_dof
```

**Step 3: 验证CMG数据加载（30分钟）**
```python
# 确认CMG输出的29 DOF顺序
# 如果需要，编写映射函数
# 测试motion_lib能否正确加载
```

**Step 4: 重新开始训练（立即）**
```bash
bash train_teacher.sh g1_cmg_29dof cuda:0
```

---

## 为什么现在修改最合适

### 1. TWIST正在训练中
- 还没有训练好的模型需要保留
- 现在改配置 = 从头训练29 DOF
- 晚改 = 浪费23 DOF的训练时间

### 2. 避免双重转换
```
错误方案：
CMG(29) → 裁剪(23) → TWIST训练 → 补充(29) → 实机
              ↑ 丢失信息        ↑ 粗糙填充

正确方案：
CMG(29) → TWIST训练(29) → 实机(29)
         ↑ 完整学习
```

### 3. 残差学习的正确姿势
```python
# 方案A（不推荐）：维度不匹配
cmg_ref = load_cmg()          # (29,)
cmg_ref_23 = cmg_ref[:23]     # 丢弃手腕
obs = build_obs(cmg_ref_23)   # 参考维度23
residual = policy(obs)        # 输出23
# 部署时手腕怎么办？固定值，学不到

# 方案B（推荐）：端到端
cmg_ref = load_cmg()          # (29,)
obs = build_obs(cmg_ref)      # 参考维度29
residual = policy(obs)        # 输出29
final = cmg_ref + residual    # (29,) 完整动作
```

---

## 潜在问题与解决

### Q1: CMG关节顺序和URDF不一致？

**检查方法：**
```python
# 打印CMG关节名
import numpy as np
data = np.load('autoregressive_motion.npz')
print(data.get('dof_names', 'No names'))

# 打印URDF关节名
# 从前面的grep结果已知
```

**解决：** 编写一次性映射函数

### Q2: 手腕关节不稳定？

**解决：**
- 调整PD增益（kp=20, kd=1）
- 增加armature（0.00425）
- 奖励中降低手腕跟踪权重

### Q3: 训练变慢？

**解决：**
- 29 DOF vs 23 DOF计算量增加不明显
- 可能需要略微增加训练时间（+10-20%）

---

## 对比总结表

| 维度 | 方案A (23 DOF) | 方案B (29 DOF) |
|------|----------------|----------------|
| **CMG输出** | 29 → 裁剪为23 | 29 → 直接使用 |
| **TWIST训练** | 23 DOF | 29 DOF |
| **实机部署** | 23+6固定 | 29完整控制 |
| **修改代价** | 无需改配置 | 改配置文件 |
| **训练时间** | 当前进度 | 重新开始 |
| **手腕控制** | 固定或规则 | 学习CMG运动 |
| **sim2real** | 训练部署不一致 | 完全一致 |
| **残差学习** | 维度不匹配 | 自然对齐 |
| **信息利用** | 丢失CMG手腕 | 充分利用 |
| **推荐指数** | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 最终推荐

**方案B：TWIST使用29 DOF训练** ← 强烈推荐

### 核心理由
1. CMG已训练好，是29 DOF的"ground truth"
2. TWIST是残差学习，应该完整学习CMG输出
3. 实机是29 DOF，训练应该和部署一致
4. **TWIST正在训练中，现在改最便宜**

### 实施优先级
1. **立即修改配置文件**（1小时内完成）
2. **验证CMG数据加载**（确保关节顺序正确）
3. **重新开始训练**（接受1-3天训练时间）
4. **长期收益**：端到端对齐，无转换损失

### 什么时候考虑方案A？
- ❌ 永远不考虑（因为TWIST正在训练，没有沉没成本）
- 除非：CMG无法提供29 DOF数据（但实际上CMG就是29 DOF）

---

## 行动建议

### 现在就做
```bash
# 1. 备份当前配置
cp legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py \
   legged_gym/legged_gym/envs/g1/g1_mimic_distill_config_23dof_backup.py

# 2. 修改为29 DOF（参考上面的代码）
vim legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py

# 3. 检查CMG数据
cd CMG_Ref
python -c "import numpy as np; d=np.load('autoregressive_motion.npz'); print(d['dof_positions'].shape)"

# 4. 重新训练
bash train_teacher.sh g1_cmg_29dof cuda:0
```

### 预期结果
- 训练时间：1-3天（和23 DOF差不多）
- 性能：更好（因为完整利用CMG）
- 部署：无缝对接实机29 DOF

**不要犹豫，现在就改！越早改，浪费的训练时间越少。**
