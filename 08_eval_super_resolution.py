import sys
import os
import torch
import numpy as np
import nibabel as nib
import math
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from models import mar, vae

# ================= 配置区 =================
device = "cuda" if torch.cuda.is_available() else "cpu"
vae_path = "vqgan/stage1.ckpt"

# ⚠️ 指向你 1250 轮或最新轮次的权重
checkpoint_path = "output_run_64_patch1/checkpoint-last.pth" 

# 选一个真实的缓存文件 (作为高清 Ground Truth)
target_npz = "output_cache/BraTS2021_00002_t1.npz" 

# ================= 工具: 保存为可视化的 NIfTI =================
def save_viewable_nifti(img_data, filename):
    """将 -1~1 的浮点图转换为 0-255 的 uint8 图"""
    img = np.clip(img_data, -1, 1)
    img = (img + 1) / 2.0 * 255.0
    img = img.astype(np.uint8)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(img, affine), filename)
    print(f"💾 已保存 NIfTI: {filename}")

# ================= SR 核心函数 (Top-K 绝对安全版) =================
def super_resolution_sampling(model, gt_tokens, fixed_mask, num_iter=64, temperature=1.0, cfg=1.0):
    bsz, seq_len, embed_dim = gt_tokens.shape
    
    tokens = gt_tokens.clone()
    tokens[fixed_mask.bool()] = 0 
    mask = fixed_mask.clone()
    
    num_unknowns = int(fixed_mask[0].sum().item())
    print(f"🎨 开始超分辨率重建 (生成缺失的 {num_unknowns/seq_len:.2%} 高频特征)...")
    
    indices = list(range(num_iter))
    
    for step in indices:
        cur_tokens = tokens.clone()
        class_embedding = model.fake_latent.repeat(bsz, 1)

        if cfg != 1.0:
            tokens_in = torch.cat([tokens, tokens], dim=0)
            class_embedding = torch.cat([class_embedding, model.fake_latent.repeat(bsz, 1)], dim=0)
            mask_in = torch.cat([mask, mask], dim=0)
        else:
            tokens_in, mask_in = tokens, mask

        x = model.forward_mae_encoder(tokens_in, mask_in, class_embedding)
        z = model.forward_mae_decoder(x, mask_in)

        # Top-K 掩码退火调度 (完美防崩溃)
        mask_ratio_step = np.cos(math.pi / 2. * (step + 1) / num_iter)
        current_mask_len = int(np.floor(num_unknowns * mask_ratio_step))
        
        mask_next = torch.zeros_like(mask)
        if current_mask_len > 0:
            random_scores = torch.rand((bsz, seq_len), device=device)
            random_scores[mask == 0] = -1.0 # 已知锚点绝对不遮挡
            _, topk_indices = torch.topk(random_scores, current_mask_len, dim=1)
            mask_next.scatter_(1, topk_indices, 1.0)
            
        mask_to_pred = torch.logical_xor(mask.bool(), mask_next.bool())
        mask = mask_next

        if cfg != 1.0:
            mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

        if mask_to_pred.sum() == 0:
            continue

        z_target = z[mask_to_pred.nonzero(as_tuple=True)]
        cfg_iter = 1 + (cfg - 1) * (seq_len - current_mask_len) / seq_len
        
        sampled_token_latent = model.diffloss.sample(z_target, temperature, cfg_iter)
        
        # 🔥 完美贴合你真实数据的截断范围，防止任何极端噪点
        sampled_token_latent = torch.clamp(sampled_token_latent, -2.0, 2.0)

        if cfg != 1.0:
            sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
            mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

        cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
        # 强制保护低分辨率的“锚点”不被改变
        cur_tokens[~fixed_mask.bool()] = gt_tokens[~fixed_mask.bool()]
        tokens = cur_tokens.clone()

    return model.unpatchify(tokens)

# ================= 工具: 生成 3x3 论文级对比概览图 =================
def save_summary_plot(gt_vol, masked_vol, pred_vol, filename):
    def to_visual_uint8(vol):
        vol = np.clip(vol, -1, 1)
        vol = (vol + 1) / 2.0 * 255.0
        return vol.astype(np.uint8)

    vols = [to_visual_uint8(gt_vol), to_visual_uint8(masked_vol), to_visual_uint8(pred_vol)]
    row_names = ['HR Ground Truth\n(64x64x64)', 'LR Input\n(Simulated 32x32x32)', 'SR Reconstruction\n(Generated 64x64x64)']
    
    d_mid, h_mid, w_mid = vols[0].shape[0] // 2, vols[0].shape[1] // 2, vols[0].shape[2] // 2

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.3) # 增加行间距容纳多行标题

    for i in range(3): 
        current_vol = vols[i]
        slices = [current_vol[d_mid, :, :], current_vol[:, h_mid, :], current_vol[:, :, w_mid]]
        
        for j in range(3): 
            ax = axes[i, j]
            ax.imshow(slices[j], cmap='gray', vmin=0, vmax=255, origin='lower')
            if i == 0:
                ax.set_title(['Axial', 'Coronal', 'Sagittal'][j], fontsize=14, fontweight='bold')
            if j == 0:
                ax.set_ylabel(row_names[i], fontsize=12, fontweight='bold', labelpad=15)
            ax.set_xticks([]); ax.set_yticks([])

    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"🖼️ 超分辨率概览图已保存: {filename}")

def main():
    # 1. 初始化 VAE
    ddconfig = {
        "double_z": False, "z_channels": 4, "resolution": 64, "in_channels": 1, "out_ch": 1, 
        "ch": 64, "num_groups": 32, "ch_mult": [1, 1, 2], "num_res_blocks": 1, "attn_resolutions": [], "dropout": 0.0
    }
    vae_model = vae.AutoencoderKL(ddconfig, 8192, 4)
    if os.path.exists(vae_path):
        sd = torch.load(vae_path, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        try:
            vae_model.load_state_dict(sd, strict=True)
            print("✅ VAE 权重加载完毕。")
        except RuntimeError:
            vae_model.load_state_dict(sd, strict=False)
    vae_model.to(device).eval()

    # 2. 初始化 MAR
    model = mar.mar_base(
        img_size=64, vae_stride=4, vae_embed_dim=4, patch_size=1, 
        num_sampling_steps="100", diffloss_d=12, diffloss_w=1536
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_ema"] if "model_ema" in checkpoint else checkpoint["model"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    try:
        model.load_state_dict(state_dict, strict=True) 
        print("✅ MAR 核心超分权重加载完美！")
    except RuntimeError as e:
        print("🚨 模型结构不匹配！")
        return 
    model.eval()

    # 3. 准备数据
    if not os.path.exists(target_npz):
        print(f"❌ 找不到缓存: {target_npz}")
        return
        
    print(f"📥 读取高分辨率(HR)样本: {target_npz}")
    data = np.load(target_npz)
    z_gt = torch.from_numpy(data['moments']).float().to(device).unsqueeze(0) 
    
    # 保持原汁原味的数据分布
    scale_factor = 2.6
    z_gt = z_gt * scale_factor

    # 4. 🌟 制造超分辨率专属的 3D 均匀网格 Mask (模拟 32x32x32 降采样)
    gt_tokens = model.patchify(z_gt) 
    bsz, L, D = gt_tokens.shape
    
    # 潜变量的网格尺寸 (4096 的立方根是 16)
    grid_size = int(round(L ** (1/3)))
    
    # 创建 3D 掩码，默认全部遮挡 (1.0)
    mask_3d = torch.ones((grid_size, grid_size, grid_size), device=device)
    
    # 核心超分逻辑：在 Z, Y, X 三个维度上，每隔一步保留一个 Token (设为 0.0)
    # 这相当于物理分辨率从 64 缩小到了 32，保留了极其纯净的低分辨率锚点
    mask_3d[::2, ::2, ::2] = 0.0
    
    # 展平回 Sequence 格式
    fixed_mask = mask_3d.view(1, L).expand(bsz, -1)
    
    mask_ratio_actual = fixed_mask[0].mean().item()
    print(f"👺 物理等效: 3D网格降采样 (保留了 1/8 的锚点), 需重建区域: {mask_ratio_actual:.2%}")

    # 5. 执行自回归超分辨重建
    with torch.no_grad():
        inpainted_z = super_resolution_sampling(model, gt_tokens, fixed_mask, num_iter=64)
        
        masked_tokens = gt_tokens.clone()
        masked_tokens[fixed_mask.bool()] = 0 
        masked_input_z = model.unpatchify(masked_tokens)

        print("🎨 正在通过 VAE 渲染回物理空间...")
        def decode_with_quant(z_latent):
            z = z_latent / scale_factor
            if hasattr(vae_model, 'post_quant_conv'):
                z = vae_model.post_quant_conv(z)
            recon = vae_model.decoder(z)
            return recon[0, 0].cpu().numpy()
        
        img_gt = decode_with_quant(z_gt)          # 高清原图 64
        img_masked = decode_with_quant(masked_input_z) # 稀疏网格图 (模拟低配 32)
        img_pred = decode_with_quant(inpainted_z)      # 超分重建图 64

    # 6. 保存与计算指标
    save_viewable_nifti(img_gt, "08_eval_SR_01_HR_GroundTruth.nii.gz")
    save_viewable_nifti(img_masked, "08_eval_SR_02_LR_Input.nii.gz") 
    save_viewable_nifti(img_pred, "08_eval_SR_03_Reconstructed.nii.gz")
    
    mse = np.mean((img_gt - img_pred) ** 2)
    summary_file = "08_eval_SR_summary_plot.png"
    save_summary_plot(img_gt, img_masked, img_pred, summary_file)
    
    val_psnr = psnr(img_gt, img_pred, data_range=255)
    val_ssim = ssim(img_gt, img_pred, data_range=255, win_size=3, channel_axis=None)

    print("\n" + "="*40)
    print(f"📈 超分辨率 (32^3 -> 64^3) 评估指标:")
    print(f"   MSE  (越低越好): {mse:.4f}")
    print(f"   PSNR (越高越好): {val_psnr:.2f} dB")
    print(f"   SSIM (越高越好): {val_ssim:.4f}")
    print("="*40)
    print("✅ 完成！快去验收你为毕业论文定制的超分对比图吧！")

if __name__ == "__main__":
    main()