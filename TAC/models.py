import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, in_ch: int, n_users: int = 7, n_tasks: int = 7):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, padding=3), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> [B,128,1]
        )
        self.head_user   = nn.Linear(128, n_users)
        self.head_task   = nn.Linear(128, n_tasks)
        self.head_attack = nn.Linear(128, 1)

    def forward(self, x):  # x: [B,C,W]
        h = self.fe(x).squeeze(-1)  # [B,128]
        return {
            "logits_user":   self.head_user(h),
            "logits_task":   self.head_task(h),
            "logits_attack": self.head_attack(h).squeeze(-1),
        }
