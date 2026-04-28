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

# ⚠️ 注意：请确保这里指向你最新 1250 轮的权重目录！
checkpoint_path = "output_run_64_patch1/checkpoint-last.pth" 

# 选一个真实的缓存文件
target_npz = "output_cache/BraTS2021_00002_t1.npz" 

# 🌟 论文专属旋钮：你想让画面多大比例变成“随机弹孔”？(0.5 = 遮挡 50%)
MASK_RATIO = 0.5 

# ================= 工具: 保存为可视化的 NIfTI =================
def save_viewable_nifti(img_data, filename):
    """将 -1~1 的浮点图转换为 0-255 的 uint8 图，确保软件能正常显示"""
    img = np.clip(img_data, -1, 1)
    img = (img + 1) / 2.0 * 255.0
    img = img.astype(np.uint8)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(img, affine), filename)
    print(f"💾 已保存: {filename}")

# ================= Inpainting 核心函数 (Top-K 绝对安全版) =================
def inpainting_sampling(model, gt_tokens, fixed_mask, num_iter=64, temperature=1.0, cfg=1.0):
    bsz, seq_len, embed_dim = gt_tokens.shape
    
    tokens = gt_tokens.clone()
    tokens[fixed_mask.bool()] = 0 
    mask = fixed_mask.clone()
    
    num_unknowns = int(fixed_mask[0].sum().item())
    print(f"🎨 开始修补... 缺失比例: {num_unknowns/seq_len:.2%}")
    
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

        # Top-K 掩码退火调度 (防崩溃核心)
        mask_ratio_step = np.cos(math.pi / 2. * (step + 1) / num_iter)
        current_mask_len = int(np.floor(num_unknowns * mask_ratio_step))
        
        mask_next = torch.zeros_like(mask)
        if current_mask_len > 0:
            random_scores = torch.rand((bsz, seq_len), device=device)
            random_scores[mask == 0] = -1.0
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
        
        # 🔥 完美贴合你真实潜变量的截断范围 (-2.0 到 2.0)
        sampled_token_latent = torch.clamp(sampled_token_latent, -2.0, 2.0)

        if cfg != 1.0:
            sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
            mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

        cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
        cur_tokens[~fixed_mask.bool()] = gt_tokens[~fixed_mask.bool()]
        tokens = cur_tokens.clone()

    return model.unpatchify(tokens)

# ================= 工具: 生成 3x3 对比概览图 =================
def save_summary_plot(gt_vol, masked_vol, pred_vol, filename):
    def to_visual_uint8(vol):
        vol = np.clip(vol, -1, 1)
        vol = (vol + 1) / 2.0 * 255.0
        return vol.astype(np.uint8)

    vols = [to_visual_uint8(gt_vol), to_visual_uint8(masked_vol), to_visual_uint8(pred_vol)]
    row_names = ['Ground Truth', f'Masked ({MASK_RATIO*100:.0f}%)', 'Inpainted Output']
    
    d_mid, h_mid, w_mid = vols[0].shape[0] // 2, vols[0].shape[1] // 2, vols[0].shape[2] // 2

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    for i in range(3): 
        current_vol = vols[i]
        slices = [current_vol[d_mid, :, :], current_vol[:, h_mid, :], current_vol[:, :, w_mid]]
        
        for j in range(3): 
            ax = axes[i, j]
            ax.imshow(slices[j], cmap='gray', vmin=0, vmax=255, origin='lower')
            if i == 0:
                ax.set_title(['Axial', 'Coronal', 'Sagittal'][j], fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(row_names[i], fontsize=12, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"🖼️ 概览图已保存: {filename}")

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
            print("✅ VAE 权重加载完美。")
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
        print("✅ MAR 模型权重 (1250 Epochs) 加载完美！")
    except RuntimeError as e:
        print("🚨 严重错误：模型结构不匹配！")
        return 
    model.eval()

    # 3. 准备数据
    if not os.path.exists(target_npz):
        print(f"❌ 找不到缓存: {target_npz}")
        return
        
    print(f"📥 读取真实样本: {target_npz}")
    data = np.load(target_npz)
    z_gt = torch.from_numpy(data['moments']).float().to(device).unsqueeze(0) 
    
    # 🔥🔥🔥 核心修复：还原比例设定为 1.0 🔥🔥🔥
    scale_factor = 1.0
    z_gt = z_gt * scale_factor

    # 4. 制造随机 Mask
    gt_tokens = model.patchify(z_gt) 
    bsz, L, D = gt_tokens.shape
    
    print(f"👺 遮挡模式: 纯随机散点遮挡，比例 {MASK_RATIO*100}%")
    noise = torch.rand(bsz, L, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    len_keep = int(L * (1 - MASK_RATIO)) 
    mask_tokens = torch.ones([bsz, L], device=device)
    mask_tokens[:, :len_keep] = 0
    fixed_mask = torch.gather(mask_tokens, dim=1, index=ids_restore)

    # 5. 执行修补
    with torch.no_grad():
        inpainted_z = inpainting_sampling(model, gt_tokens, fixed_mask, num_iter=64)
        
        masked_tokens = gt_tokens.clone()
        masked_tokens[fixed_mask.bool()] = 0 
        masked_input_z = model.unpatchify(masked_tokens)

        print("🎨 正在解码...")
        def decode_with_quant(z_latent):
            z = z_latent / scale_factor
            if hasattr(vae_model, 'post_quant_conv'):
                z = vae_model.post_quant_conv(z)
            recon = vae_model.decoder(z)
            return recon[0, 0].cpu().numpy()
        
        img_gt = decode_with_quant(z_gt)
        img_masked = decode_with_quant(masked_input_z)
        img_pred = decode_with_quant(inpainted_z)

    # 6. 保存与计算指标
    save_viewable_nifti(img_gt, "08_eval_01_GroundTruth.nii.gz")
    save_viewable_nifti(img_masked, "08_eval_02_MaskedInput.nii.gz") 
    save_viewable_nifti(img_pred, "08_eval_03_Inpainted.nii.gz")
    
    mse = np.mean((img_gt - img_pred) ** 2)
    summary_file = "08_eval_summary_plot.png"
    save_summary_plot(img_gt, img_masked, img_pred, summary_file)
    
    val_psnr = psnr(img_gt, img_pred, data_range=255)
    val_ssim = ssim(img_gt, img_pred, data_range=255, win_size=3, channel_axis=None)

    print(f"📈 评估指标 (遮挡率: {MASK_RATIO*100}%):")
    print(f"   MSE  (越低越好): {mse:.4f}")
    print(f"   PSNR (越高越好): {val_psnr:.2f} dB")
    print(f"   SSIM (越高越好): {val_ssim:.4f}")
    print("✅ 完成！快去验收你 1250 轮炼丹的心血吧！")

if __name__ == "__main__":
    main()