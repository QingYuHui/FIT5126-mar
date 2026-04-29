"""
Purpose:
    Print the raw keys and tensor dimensions inside the stage-1 VQGAN/VAE
    checkpoint. This is mainly a debugging script for checking whether the
    checkpoint contains 3D convolution weights and decoder parameters.

Suggested filename:
    01_inspect_stage1_checkpoint_keys.py
"""

import torch
import sys
import os

# 你的权重路径
CKPT_PATH = r"vqgan/stage1.ckpt"

def inspect_checkpoint():
    if not os.path.exists(CKPT_PATH):
        print(f"❌ 找不到文件: {CKPT_PATH}")
        return

    print(f"📂 正在加载: {CKPT_PATH} ...")
    try:
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 1. 找到 state_dict
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        print("✅ 发现 'state_dict' 键")
    elif "model" in ckpt:
        sd = ckpt["model"]
        print("✅ 发现 'model' 键")
    else:
        sd = ckpt
        print("ℹ️ 直接使用 checkpoint 作为 state_dict")

    # 2. 打印前 20 个 Key 来观察前缀
    keys = list(sd.keys())
    print(f"\n🔑 总共有 {len(keys)} 个参数张量。")
    print("-" * 40)
    print("前 20 个 Key (请观察是否有前缀，如 'module.' 或 'autoencoder.'):")
    for k in keys[:20]:
        print(f"  {k}")
    print("-" * 40)

    # 3. 维度检查 (验证是否真的是 3D)
    # 尝试找一个卷积层
    conv_keys = [k for k in keys if "conv" in k and "weight" in k]
    if conv_keys:
        sample_key = conv_keys[0]
        shape = sd[sample_key].shape
        print(f"📐 维度抽查 [{sample_key}]:")
        print(f"   Shape: {shape}")
        if len(shape) == 5: # [Out, In, D, H, W]
            print("   ✅ 确认: 这是 3D 卷积权重")
        elif len(shape) == 4: # [Out, In, H, W]
            print("   ⚠️ 警告: 这是 2D 卷积权重! (如果你认为是 3D，可能拿错文件了)")
        else:
            print(f"   ℹ️ 未知维度: {len(shape)}D")
    
    # 4. Decoder 关键层检查
    print("\n🔍 Decoder 存在性检查:")
    decoder_keys = [k for k in keys if "decoder" in k]
    if len(decoder_keys) == 0:
        print("   ❌ 严重警告: 权重文件中没有包含 'decoder' 字样的 Key！")
        print("   可能原因: 这是一个只有 Encoder 的权重，或者是命名完全不同的模型。")
    else:
        print(f"   ✅ 包含 {len(decoder_keys)} 个 Decoder 参数。")
        print(f"   示例: {decoder_keys[0]}")

if __name__ == "__main__":
    inspect_checkpoint()
