# TWIST训练错误的正确解决方案

## 问题分析（重新理解）

你是对的！我之前误解了项目架构。

### 正确的理解

**训练阶段1（当前冷启动阶段）：**
```
速度命令(vx, vy, yaw) → CMG生成器 → 参考动作序列(29 DOF)
                                           ↓
                                    TWIST环境追踪学习
                                           ↓
                              策略输出 = 参考动作 + 残差修正
```

**关键点：**
- CMG已经训练好，作为frozen参考生成器
- 不需要mocap数据（那是阶段2遥操作才需要的）
- TWIST学习的是**基于速度命令的locomotion**，不是遥操作

## 当前代码问题

### 问题所在
`legged_gym/legged_gym/envs/base/humanoid_mimic.py` 第66行：

```python
def _load_motions(self):
    self._motion_lib = MotionLib(motion_file=self.cfg.motion.motion_file, device=self.device)
    return
```

**问题：** 
- 环境试图从pkl文件加载mocap数据
- 但冷启动阶段应该使用CMG生成动作
- 服务器上没有mocap数据（也不需要）

### 已实现但未集成的工具

`CMG_Ref/utils/cmg_motion_generator.py` 已经实现了：
- `CMGMotionGenerator` 类
- 支持4096并行环境
- 双模式运行（预生成/实时）
- 但**环境还没有使用它**

## 正确的解决方案

### 选项1：创建CMG版本的MotionLib（推荐）

创建 `pose/pose/utils/motion_lib_cmg.py`，包装CMGMotionGenerator：

```python
class MotionLibCMG:
    """使用CMG生成参考动作的MotionLib替代品"""
    
    def __init__(self, cmg_model_path, cmg_data_path, device, num_envs, mode='pregenerated'):
        from CMG_Ref.utils.cmg_motion_generator import CMGMotionGenerator
        
        self.generator = CMGMotionGenerator(
            model_path=cmg_model_path,
            data_path=cmg_data_path,
            num_envs=num_envs,
            mode=mode,
            device=device
        )
    
    def reset(self, env_ids, commands):
        """根据速度命令生成新的参考轨迹"""
        self.generator.reset(env_ids, commands)
    
    def get_motion_state(self, time_offsets):
        """获取当前时刻的参考状态"""
        ref_dof_pos, ref_dof_vel = self.generator.get_motion()
        # 返回与MotionLib兼容的格式
        return ref_dof_pos, ref_dof_vel
```

### 选项2：修改环境直接使用CMG

在 `g1_mimic_distill.py` 中：

```python
def __init__(self, cfg, ...):
    # 不调用父类的_load_motions，直接初始化CMG
    self.use_cmg = cfg.motion.use_cmg  # 新增配置项
    
    if self.use_cmg:
        from CMG_Ref.utils.cmg_motion_generator import CMGMotionGenerator
        self.motion_generator = CMGMotionGenerator(...)
    else:
        super().__init__(...)  # 使用传统mocap

def reset_idx(self, env_ids):
    if self.use_cmg:
        # 采样速度命令
        commands = self.sample_commands(env_ids)
        # 让CMG生成参考轨迹
        self.motion_generator.reset(env_ids, commands)
    else:
        super().reset_idx(env_ids)
```

## 立即可行的临时方案

### 方案A：使用已有的CMG集成工具

如果CMG模型已训练好：

```bash
cd CMG_Ref

# 1. 使用CMGMotionGenerator批量生成训练数据
python -c "
from utils.cmg_motion_generator import CMGMotionGenerator
from utils.command_sampler import CommandSampler
import torch

# 初始化生成器
generator = CMGMotionGenerator(
    model_path='runs/cmg_final.pt',
    data_path='dataloader/cmg_training_data.pt',
    num_envs=1000,  # 生成1000条轨迹
    mode='pregenerated',
    preload_duration=500
)

# 采样命令
sampler = CommandSampler()
commands = sampler.sample_batch(1000, strategy='stage1_basic')

# 生成并保存
generator.reset(commands=commands)
# ... 保存为pkl格式 ...
"
```

### 方案B：创建最小MotionLibCMG

快速实现最小版本：

1. 创建 `pose/pose/utils/motion_lib_cmg.py`
2. 实现与MotionLib相同的接口
3. 内部调用CMGMotionGenerator
4. 修改配置使用CMG版本

### 方案C：跳过MotionLib（最快）

修改 `humanoid_mimic.py`：

```python
def _load_motions(self):
    # 临时跳过，使用随机参考
    if hasattr(self.cfg.motion, 'use_cmg') and self.cfg.motion.use_cmg:
        print("[INFO] 使用CMG模式，跳过MotionLib加载")
        self._motion_lib = None  # 标记为CMG模式
        return
    
    # 原有逻辑
    self._motion_lib = MotionLib(...)
```

## 推荐实施步骤

### 第1步：验证CMG模型可用

```bash
cd CMG_Ref
python -c "
import torch
from module.cmg import CMG

# 检查模型文件
model_path = 'runs/cmg_20260123_194851/cmg_final.pt'
data_path = 'dataloader/cmg_training_data.pt'

checkpoint = torch.load(model_path, weights_only=False)
print('✓ CMG模型加载成功')
print(f'  Epoch: {checkpoint.get(\"epoch\", \"N/A\")}')
"
```

### 第2步：创建MotionLibCMG（今天完成）

实现完整的CMG适配器，保持与MotionLib接口兼容

### 第3步：更新配置（今天完成）

```python
# g1_mimic_distill_config.py
class motion(HumanoidMimicCfg.motion):
    use_cmg = True  # 新增：使用CMG而不是mocap
    cmg_model_path = f"{LEGGED_GYM_ROOT_DIR}/../CMG_Ref/runs/cmg_final.pt"
    cmg_data_path = f"{LEGGED_GYM_ROOT_DIR}/../CMG_Ref/dataloader/cmg_training_data.pt"
    cmg_mode = 'pregenerated'  # 或 'realtime'
```

### 第4步：测试训练

```bash
export TWIST_USE_CMG=1  # 标记使用CMG模式
bash train_teacher.sh test_cmg cuda:0
```

## 为什么之前的方案不对

**之前的方案（设置TWIST_MOTION_DATA_ROOT）：**
- ❌ 试图解决"文件路径不存在"问题
- ❌ 假设需要mocap数据
- ❌ 没有理解冷启动阶段应该用CMG

**正确的理解：**
- ✅ 冷启动阶段**不需要mocap数据**
- ✅ 应该使用CMG生成参考动作
- ✅ MotionLib应该被CMG替代（或包装）

## 下一步行动

我可以立即帮你：

1. **创建MotionLibCMG** - 包装CMGMotionGenerator，提供MotionLib兼容接口
2. **修改环境配置** - 添加use_cmg开关
3. **更新环境代码** - 支持CMG模式
4. **创建测试脚本** - 验证CMG集成正常工作

需要我继续实施吗？
