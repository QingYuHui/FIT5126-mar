# 检查训练存档，确认当前训练进度


import torch
# 你的存档路径
ckpt_path = "output_run_64_patch1/checkpoint-last.pth" 
ckpt = torch.load(ckpt_path, map_location='cpu')
print(f"当前存档保存于 Epoch: {ckpt['epoch']}")
print(f"下一次启动将从 Epoch: {ckpt['epoch'] + 1} 开始")