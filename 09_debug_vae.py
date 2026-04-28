import sys
import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ================= 配置 =================
# 1. 指向你生成的某一个缓存文件
NPZ_PATH = r"output_cache\class0\BraTS2021_00002_t1.npz" 
# 2. VAE 权重
VAE_PATH = r"vqgan\stage1.ckpt"

# =======================================

from models import vae

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔍 正在检查文件: {NPZ_PATH}")

    # -------------------------------------------
    # 步骤 1: 检查 .npz 里的 Latent 到底长什么样
    # -------------------------------------------
    if not os.path.exists(NPZ_PATH):
        print("❌ 找不到 .npz 文件，请检查路径")
        return

    data = np.load(NPZ_PATH)
    z = data['moments'] # [4, 16, 16, 16] (如果是 64 尺寸)
    
    print(f"📊 Latent Shape: {z.shape}")
    print(f"📊 Latent 数值范围: Min={z.min():.4f}, Max={z.max():.4f}, Mean={z.mean():.4f}")

    # 绘图：Latent 的中间切片
    # 如果这里看着像乱码/纯色，说明 main_cache.py 生成时就错了（没归一化）
    z_slice = z[0, z.shape[2]//2, :, :] # 取第0个通道的中间切片
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(z_slice, cmap='viridis')
    plt.colorbar()
    plt.title("Latent Representation (Channel 0)")
    
    # -------------------------------------------
    # 步骤 2: 正确初始化 VAE 并尝试解码
    # -------------------------------------------
    ddconfig = {
        "double_z": False, "z_channels": 4, "resolution": 64, "in_channels": 1, "out_ch": 1, 
        "ch": 64, "num_groups": 32, "ch_mult": [1, 1, 2], "num_res_blocks": 1, 
        "attn_resolutions": [], "dropout": 0.0
    }
    
    model = vae.AutoencoderKL(ddconfig, 8192, 4).to(device)
    
    # 加载权重
    if os.path.exists(VAE_PATH):
        sd = torch.load(VAE_PATH, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
        print("✅ VAE 权重已加载")
    else:
        print("❌ 找不到 VAE 权重！解码出来肯定是噪声。")
        return

    model.eval()

    # 准备 Latent
    z_tensor = torch.from_numpy(z).unsqueeze(0).to(device).float() # [1, 4, 16, 16, 16]
    
    with torch.no_grad():
        # 🔥 关键步骤：Post Quant Conv 🔥
        # VQGAN 必须经过这一层才能把 latent 映射回 decoder 认识的维度
        if hasattr(model, 'post_quant_conv'):
            print("⚙️ 执行 post_quant_conv...")
            quant = model.post_quant_conv(z_tensor)
        else:
            print("⚠️ 警告：模型没有 post_quant_conv 层，解码可能异常！")
            quant = z_tensor
            
        # 解码
        recon = model.decoder(quant) # [1, 1, 64, 64, 64]
    
    # -------------------------------------------
    # 步骤 3: 检查输出并保存
    # -------------------------------------------
    img = recon[0, 0].cpu().numpy()
    print(f"🖼️ 解码后图像数值范围: Min={img.min():.4f}, Max={img.max():.4f}")

    # 🎨 绘图：解码后的中间切片
    plt.subplot(1, 2, 2)
    plt.imshow(img[img.shape[0]//2, :, :], cmap='gray')
    plt.colorbar()
    plt.title("Decoded Image (GT)")
    plt.savefig("debug_check_gt.png")
    print("✅ 诊断图已保存为 debug_check_gt.png，请打开查看！")

    # -------------------------------------------
    # 步骤 4: 保存 NIfTI (修正显示范围)
    # -------------------------------------------
    # 很多医学软件如果读到 -1~1 的 float 数据会显示全黑
    # 我们把它拉伸到 0~255 的整数
    img_norm = np.clip(img, -1, 1)
    img_norm = (img_norm + 1) / 2.0 * 255.0
    img_uint8 = img_norm.astype(np.uint8)

    affine = np.eye(4)
    nib.save(nib.Nifti1Image(img_uint8, affine), "debug_gt_uint8.nii.gz")
    print("✅ 已保存 debug_gt_uint8.nii.gz (已转为0-255范围，请用软件打开此文件)")

if __name__ == "__main__":
    main()