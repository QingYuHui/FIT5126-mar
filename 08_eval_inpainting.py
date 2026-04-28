# 给模型半个脑子，让它画出另外半个

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
# 确保这里指向你训练好的 Checkpoint
checkpoint_path = "output_run_64_patch1/checkpoint-last.pth"
# 选一个真实的缓存文件
target_npz = "output_cache/BraTS2021_00002_t1.npz" 

# ================= 工具: 保存为可视化的 NIfTI =================
def save_viewable_nifti(img_data, filename):
    """将 -1~1 的浮点图转换为 0-255 的 uint8 图，确保软件能正常显示"""
    # 1. 截断异常值
    img = np.clip(img_data, -1, 1)
    # 2. 归一化到 0-255
    img = (img + 1) / 2.0 * 255.0
    # 3. 转整数
    img = img.astype(np.uint8)
    # 4. 保存
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(img, affine), filename)
    print(f"💾 已保存: {filename}")

# ================= Inpainting 核心函数 =================
def inpainting_sampling(model, gt_tokens, fixed_mask, num_iter=64, temperature=1.0, cfg=1.0):
    bsz, seq_len, embed_dim = gt_tokens.shape
    
    # 初始化: 已知部分填 GT，未知部分填 0
    tokens = gt_tokens.clone()
    tokens[fixed_mask.bool()] = 0 
    
    mask = fixed_mask.clone()
    orders = model.sample_orders(bsz)
    
    print(f"🎨 开始修补... 缺失比例: {fixed_mask.float().mean().item():.2%}")
    
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

        # 1. 获取上下文特征 z
        x = model.forward_mae_encoder(tokens_in, mask_in, class_embedding)
        z = model.forward_mae_decoder(x, mask_in)

        # 2. 计算 MaskGIT 调度 (只在 fixed_mask 区域内退火)
        mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
        mask_len = torch.Tensor([np.floor(seq_len * mask_ratio)]).to(device)
        
        mask_next = mar.mask_by_order(mask_len[0], orders, bsz, seq_len)
        # 关键: 确保 mask 不会覆盖已知区域
        mask_next = torch.logical_and(mask_next.bool(), fixed_mask.bool()).float()
        
        mask_to_pred = torch.logical_xor(mask.bool(), mask_next.bool())
        mask = mask_next

        if cfg != 1.0:
            mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

        if mask_to_pred.sum() == 0:
            continue

        # 3. Diffusion 去噪 (Sample)
        z_target = z[mask_to_pred.nonzero(as_tuple=True)]
        cfg_iter = 1 + (cfg - 1) * (seq_len - mask_len[0]) / seq_len
        
        sampled_token_latent = model.diffloss.sample(z_target, temperature, cfg_iter)
        
        # 🔥🔥🔥【核心修复】强制截断数值，防止爆炸 🔥🔥🔥
        # 我们的数据 scale 之后大约在 -3 到 3 之间。
        # 给一点余量，限制在 -5 到 5。任何超过这个范围的都是错误累积，必须切掉！
        sampled_token_latent = torch.clamp(sampled_token_latent, -5.0, 5.0)

        if cfg != 1.0:
            sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
            mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

        # 4. 更新 Token
        cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
        
        # 5. 强制回填 GT (保证已知区域不被修改)
        cur_tokens[~fixed_mask.bool()] = gt_tokens[~fixed_mask.bool()]
        
        tokens = cur_tokens.clone()

    return model.unpatchify(tokens)

# ================= 工具: 生成 3x3 对比概览图 =================
def save_summary_plot(gt_vol, masked_vol, pred_vol, filename):
    """
    生成一个 3x3 的对比图:
    行: Ground Truth, Masked Input, Inpainted Output
    列: Axial(横断), Coronal(冠状), Sagittal(矢状) 切片
    """
    # 内部辅助函数：将数据转为可视化的 uint8
    def to_visual_uint8(vol):
        vol = np.clip(vol, -1, 1)
        vol = (vol + 1) / 2.0 * 255.0
        return vol.astype(np.uint8)

    # 1. 准备数据
    vols = [to_visual_uint8(gt_vol), to_visual_uint8(masked_vol), to_visual_uint8(pred_vol)]
    row_names = ['Ground Truth', 'Masked Input', 'Inpainted Output']
    
    # 获取中心切片索引 (假设形状是 D, H, W)
    d, h, w = vols[0].shape
    d_mid, h_mid, w_mid = d // 2, h // 2, w // 2

    # 2. 创建画布
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    for i in range(3): # 遍历三行 (GT, Masked, Pred)
        current_vol = vols[i]
        
        # 提取三个方向的切片
        # 注意：根据数据方向，可能需要转置 (.T) 或翻转以符合观看习惯
        # 这里假设数据是 (Depth, Height, Width)
        slices = [
            current_vol[d_mid, :, :],       # Axial (横断面)
            current_vol[:, h_mid, :],       # Coronal (冠状面)
            current_vol[:, :, w_mid]        # Sagittal (矢状面)
        ]
        
        for j in range(3): # 遍历三列 (Axial, Coronal, Sagittal)
            ax = axes[i, j]
            # 使用灰度图显示，固定范围 0-255
            ax.imshow(slices[j], cmap='gray', vmin=0, vmax=255, origin='lower')
            
            # 设置标题和标签
            if i == 0:
                col_names = ['Axial', 'Coronal', 'Sagittal']
                ax.set_title(col_names[j], fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(row_names[i], fontsize=12, fontweight='bold')
            
            # 去掉坐标轴刻度
            ax.set_xticks([])
            ax.set_yticks([])

    # 3. 保存
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
        # vae_model.load_state_dict(torch.load(vae_path, map_location="cpu")["state_dict"], strict=False)
        sd = torch.load(vae_path, map_location="cpu")
        if "state_dict" in sd: 
            sd = sd["state_dict"]
        
        # 🔥🔥 改用 strict=True (在 try-catch 中) 来捕获不匹配的键 🔥🔥
        try:
            vae_model.load_state_dict(sd, strict=True)
            print("✅ VAE weights loaded PERFECTLY (Strict match).")
        except RuntimeError as e:
            print("⚠️ VAE 权重加载不完全！正在分析缺失键...")
            # 打印错误信息中提到的 missing keys
            print(e)
            # 还是强行加载，但我们要知道缺了啥
            vae_model.load_state_dict(sd, strict=False)
    vae_model.to(device).eval()

    # 2. 初始化 MAR (注意 patch_size=1)
    model = mar.mar_base(
        img_size=64, vae_stride=4, vae_embed_dim=4, patch_size=1, 
        num_sampling_steps="100", diffloss_d=12, diffloss_w=1536
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_ema"] if "model_ema" in checkpoint else checkpoint["model"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # model.load_state_dict(state_dict, strict=True)
    try:
        msg = model.load_state_dict(state_dict, strict=True) 
        print("✅ MAR 模型权重加载完美！")
    except RuntimeError as e:
        print("🚨 严重错误：模型结构不匹配！")
        print(e)
        return # 直接退出，不要往下跑了，跑了也是雪花
    model.eval()

    # 3. 准备数据
    if not os.path.exists(target_npz):
        print(f"❌ 找不到缓存: {target_npz}")
        return
        
    print(f"📥 读取真实样本: {target_npz}")
    data = np.load(target_npz)
    z_gt = torch.from_numpy(data['moments']).float().to(device).unsqueeze(0) 
    
    # 特殊处理: 放大 latent 数值
    scale_factor = 2.6
    z_gt = z_gt * scale_factor

    # 4. 制造 Mask (Patch 级别遮挡)
    gt_tokens = model.patchify(z_gt) 
    # 🧪【验证测试】立即还原，看是否和原图一致
    z_recon_check = model.unpatchify(gt_tokens)
    
    # 计算差异
    diff = (z_gt - z_recon_check).abs().max().item()
    print(f"🕵️ patchify/unpatchify 差异检查: {diff:.6f}")
    if diff > 1e-4:
        print("🚨 警报：patchify 还原失败！图像被打乱了！这就是雪花的原因！")
    else:
        print("✅ patchify 还原正常。")

    bsz, L, D = gt_tokens.shape
    
    # --- Mask 逻辑 ---
    # 我们不再在 3D 空间里猜坐标轴，直接在 Token 序列上遮挡
    # 这样能保证 Mask 的位置绝对一致
    
    # 创建全 0 mask
    fixed_mask = torch.zeros((bsz, L)).to(device)
    
    # 策略：遮住 Token 序列的后 50%
    # 这对应图像的哪一部分取决于 patchify 的顺序，但无论对应哪里，
    # 输入和输出现在一定会对齐！
    half_len = L // 2
    fixed_mask[:, half_len:] = 1.0 
    
    print(f"👺 遮挡模式: Token 序列后 50% 被遮挡")
    # ================= 🆕 修改：随机遮挡 50% (简单模式) =================
    # # 1. 生成随机噪声用于打乱
    # noise = torch.rand(bsz, L, device=device)
    # 
    # # 2. 排序得到打乱的索引
    # ids_shuffle = torch.argsort(noise, dim=1)
    # ids_restore = torch.argsort(ids_shuffle, dim=1)
    # 
    # # 3. 设定保留比例：保留 50% (即遮挡 50%)
    # # 如果想更简单，可以改成 0.7 (保留70%，只遮30%)
    # len_keep = int(L * 0.3) 
    # 
    # # 4. 制造 Mask
    # # 先假设全都要遮挡 (1)
    # mask_tokens = torch.ones([bsz, L], device=device)
    # # 把前 len_keep 个位置设为可见 (0)
    # mask_tokens[:, :len_keep] = 0
    # # 关键：把 Mask 还原回原来的像素顺序 (这样就是随机的了，而不是遮挡前一半)
    # fixed_mask = torch.gather(mask_tokens, dim=1, index=ids_restore)
    
    print(f"👺 遮挡模式: 随机遮挡 50% (Easy Mode)")

    # 5. 执行修补
    with torch.no_grad():
        # A. 生成补全后的 Latent
        inpainted_z = inpainting_sampling(model, gt_tokens, fixed_mask, num_iter=64)
        
        # 🚨 作弊模式：我不想让 MAR 猜了，我就想看看如果你拿到正确答案，能不能画出好图？
        # print("🚨 正在进行作弊测试：直接使用 Ground Truth...")
        # inpainted_z = model.unpatchify(gt_tokens)
        
        # B. 生成"被遮挡的输入" (用于对比展示)
        # 关键修改：直接用 gt_tokens 和 fixed_mask 来生成这个视图
        # 把被遮挡的部分填成 0 (或 -1/噪声)，模拟模型看到的残缺输入
        masked_tokens = gt_tokens.clone()
        # 将被遮挡的 token 设为 0 (对应 latent 里的 0 值，即灰色背景)
        masked_tokens[fixed_mask.bool()] = 0 
        masked_input_z = model.unpatchify(masked_tokens)

        # C. 解码所有图像
        # 除 scale_factor, 缩放回去
        print("🎨 正在解码...")
        # img_gt = vae_model.decoder(z_gt / scale_factor)[0, 0].cpu().numpy()
        # img_masked = vae_model.decoder(masked_input_z / scale_factor)[0, 0].cpu().numpy()
        # img_pred = vae_model.decoder(inpainted_z / scale_factor)[0, 0].cpu().numpy()
        
        # 定义一个安全的解码辅助函数
        def decode_with_quant(z_latent):
            # 1. 还原信号强度
            z = z_latent / scale_factor
            
            # 🔥 debug 打印 1: 看看还原后的数值范围正常吗？(-1 到 1 之间是比较正常的)
            print(f"DEBUG [Pre-Conv]: Min={z.min():.2f}, Max={z.max():.2f}, Mean={z.mean():.2f}")

            # 2. 执行 post_quant_conv
            if hasattr(vae_model, 'post_quant_conv'):
                print("DEBUG: ✅ 正在执行 post_quant_conv") 
                z = vae_model.post_quant_conv(z)
            else:
                print("DEBUG: ❌ 警告！模型没有 post_quant_conv 层！")

            # 🔥 debug 打印 2: 看看进解码器前的数据范围
            print(f"DEBUG [To-Decoder]: Min={z.min():.2f}, Max={z.max():.2f}")

            # 3. 解码
            recon = vae_model.decoder(z)
            return recon[0, 0].cpu().numpy()
        
        img_gt = decode_with_quant(z_gt)
        img_masked = decode_with_quant(masked_input_z)
        img_pred = decode_with_quant(inpainted_z)

    # 6. 保存
    save_viewable_nifti(img_gt, "08_eval_01_GroundTruth.nii.gz")
    save_viewable_nifti(img_masked, "08_eval_02_MaskedInput.nii.gz") # 现在这张图绝对诚实了
    save_viewable_nifti(img_pred, "08_eval_03_Inpainted.nii.gz")
    
    mse = np.mean((img_gt - img_pred) ** 2)
    print(f"📉 补全误差 (MSE): {mse:.6f}")

    print("📊 正在生成概览图...")
    summary_file = "08_eval_summary_plot.png"
    save_summary_plot(img_gt, img_masked, img_pred, summary_file)
    
    # 1. 计算 PSNR
    # data_range 必须指定，对于 uint8 是 255
    val_psnr = psnr(img_gt, img_pred, data_range=255)

    # 2. 计算 SSIM
    # win_size 必须小于图像最小边长 (64)，channel_axis=None 表示是 3D 灰度体数据
    val_ssim = ssim(img_gt, img_pred, data_range=255, win_size=3, channel_axis=None)

    print(f"📈 评估指标:")
    print(f"   MSE  (越低越好): {mse:.4f}")
    print(f"   PSNR (越高越好): {val_psnr:.2f} dB")
    print(f"   SSIM (越高越好): {val_ssim:.4f}")

    print("✅ 完成！请查看 08_eval_summary_plot.png 和 NIfTI 文件。")

if __name__ == "__main__":
    main()