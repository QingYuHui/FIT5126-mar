# Allow this copied script to be run from either the repository root or Evaluate/.
from pathlib import Path
import sys
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

# 让训练好的 AI 模型 “凭空想象” 画出一个 3D 大脑，而不需要参考任何现有的图片
# graph LR
#    A[随机初始噪声/全Mask] -->|MAR模型: sample_tokens| B(生成潜在向量 Latent z)
#    B -->|VAE模型: decode| C(解码出像素图像 Image)
#    C -->|Nibabel| D[保存为 .nii.gz 文件]

"""
Purpose:
    Generate a full 64^3 latent volume from the trained MAR model using
    unconditional sampling, decode it with the VQGAN/VAE, and save the result
    as a NIfTI file.

Suggested filename:
    07_sample_unconditional_mar_volume.py

Notes:
    This is a qualitative generation smoke test, not a super-resolution
    benchmark.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib

from models import mar, vae

device = "cuda" if torch.cuda.is_available() else "cpu"
vae_path = "vqgan/stage1.ckpt"
checkpoint_path = "output_run_64_patch1/checkpoint-last.pth"

def main():
    print(f"🚀 Device: {device}")

    # ================= 1. 初始化 VAE (使用验证过的配置) =================
    # 必须匹配 stage1.ckpt 的结构: f4 模型
    ddconfig = {
        "double_z": False,
        "z_channels": 4,
        "resolution": 64,  # 匹配你的 img_size
        "in_channels": 1,
        "out_ch": 1,
        "ch": 64,
        "num_groups": 32,
        "ch_mult": [1, 1, 2], # f4
        "num_res_blocks": 1,
        "attn_resolutions": [],
        "dropout": 0.0
    }
    
    print(f"Loading VAE from {vae_path}...")
    # 使用 8192 和 4 是为了占位，实际上 decode 不需要它们，但类初始化需要
    vae_model = vae.AutoencoderKL(ddconfig, 8192, 4)
    
    # 加载权重 (Strict=False 忽略 discriminator)
    if os.path.exists(vae_path):
        sd = torch.load(vae_path, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        vae_model.load_state_dict(sd, strict=False)
        print("✅ VAE weights loaded.")
    else:
        print("❌ VAE path not found!")
        return

    vae_model.to(device).eval()
    
    # ================= 2. 加载 MAR (修正参数) =================
    print(f"Loading MAR...")
    model = mar.mar_base(
        img_size=64,
        vae_stride=4,
        vae_embed_dim=4,
        patch_size=1,
        num_sampling_steps="100",
        diffloss_d=12,
        diffloss_w=1536
    ).to(device)

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading MAR Checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 优先加载 EMA 权重 (通常效果更好)
    if "model_ema" in checkpoint:
        print("✅ Using EMA weights (Smoother results)")
        state_dict = checkpoint["model_ema"]
    else:
        print("⚠️ Using Standard weights")
        state_dict = checkpoint["model"]

    # 去除 module. 前缀 (如果是 DDP 训练出来的)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # 加载权重
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Load Status: {msg}")
    model.eval()

    # ================= 3. 生成 (Sampling) =================
    print("⚡ Generating latent vectors...")
    with torch.no_grad():
        # sample_tokens 会自动执行 Diffusion 的逆向去噪过程
        # bsz=1: 生成 1 个大脑
        sampled_z = model.sample_tokens(bsz=1) 
        
        # 预期形状: [1, 4, 16, 16, 16] (因为 64 / 4 = 16)
        print(f"Generated Latent Shape: {sampled_z.shape}")
        
        # ================= 4. 解码 (Decode) =================
        print("🎨 Decoding with VAE...")
        
        sampled_z = sampled_z / 2.6
        
        # 标准 VQGAN 解码流程: Latent -> PostQuantConv -> Decoder
        # 你的 AutoencoderKL.decode 通常已经封装了 post_quant_conv，
        # 如果没有封装，手动调一下:
        if hasattr(vae_model, 'post_quant_conv'):
            quant = vae_model.post_quant_conv(sampled_z)
        else:
            quant = sampled_z

        generated_images = vae_model.decoder(quant)
        
        # 截断到合法范围
        generated_images = torch.clamp(generated_images, -1, 1)
        
        # [1, 1, 64, 64, 64] -> [64, 64, 64] (Numpy)
        img_3d = generated_images[0, 0, :, :, :].cpu().numpy()

    # ================= 5. 保存 NIfTI =================
    save_filename = "07_check_gen_64_f4.nii.gz"
    
    # 简单的仿射矩阵 (Identity)
    affine = np.eye(4)
    nii_img = nib.Nifti1Image(img_3d, affine)
    
    nib.save(nii_img, save_filename)
    print(f"📊 Final Image Shape: {img_3d.shape}")
    print(f"🎉 Result saved to: {os.path.abspath(save_filename)}")

if __name__ == "__main__":
    main()
