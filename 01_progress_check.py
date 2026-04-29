# 检查训练存档，确认当前训练进度


"""
Purpose:
    Inspect the latest MAR training checkpoint and print the saved epoch.
    Use this to confirm resume progress before continuing training.

Suggested filename:
    01_check_mar_training_progress.py
"""

import torch
# 你的存档路径
ckpt_path = "output_run_64_patch1/checkpoint-last.pth" 
ckpt = torch.load(ckpt_path, map_location='cpu')
print(f"当前存档保存于 Epoch: {ckpt['epoch']}")
print(f"下一次启动将从 Epoch: {ckpt['epoch'] + 1} 开始")
