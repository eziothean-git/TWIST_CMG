import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import importlib.util
from pathlib import Path

# 动态加载CMG模块
_cmg_module_cache = None

def _get_cmg_class():
    global _cmg_module_cache
    if _cmg_module_cache is None:
        cmg_ref_root = Path(__file__).parent
        module_path = str(cmg_ref_root / "module" / "cmg.py")
        spec = importlib.util.spec_from_file_location("cmg", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载CMG模块: {module_path}")
        _cmg_module_cache = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_cmg_module_cache)
    return _cmg_module_cache.CMG

CMG = None  # 将在运行时设置

class CMGTrainer:
    def __init__(
        self,
        model,  # 改为不指定类型，因为CMG是动态加载的
        lr: float,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.device = device
        
        # Scheduled sampling参数
        self.teacher_prob = 1.0
        self.teacher_prob_decay = 0.995  # 每个epoch衰减
        self.teacher_prob_min = 0.3
    
    def train_epoch(self, dataloader, use_scheduled_sampling: bool = True):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            motion_seq = batch["motion"].to(self.device)    # [batch, seq_len, motion_dim]
            command_seq = batch["command"].to(self.device)  # [batch, seq_len, command_dim]
            
            batch_size, seq_len, _ = motion_seq.shape
            loss = 0
            
            # 第一帧用ground truth
            current_motion = motion_seq[:, 0]
            
            for t in range(seq_len - 1):
                command = command_seq[:, t]
                target = motion_seq[:, t + 1]
                
                # 预测下一帧
                pred = self.model(current_motion, command)
                
                # 计算损失
                loss = loss + nn.functional.mse_loss(pred, target)
                
                # Scheduled sampling: 决定下一步用预测还是ground truth
                if use_scheduled_sampling:
                    if torch.rand(1).item() < self.teacher_prob:
                        current_motion = motion_seq[:, t + 1]  # teacher: 用ground truth
                    else:
                        current_motion = pred.detach()  # student: 用自己的预测
                else:
                    current_motion = motion_seq[:, t + 1]
            
            loss = loss / (seq_len - 1)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # 衰减teacher probability
        self.teacher_prob = max(self.teacher_prob_min, self.teacher_prob * self.teacher_prob_decay)
        
        return total_loss / len(dataloader)