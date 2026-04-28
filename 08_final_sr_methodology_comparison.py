import sys
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from nibabel.processing import resample_from_to

# ================= 配置区 =================
# 1. 🌟 绝对对齐的 Ground Truth (由 SR eval 脚本生成的那个！)
aligned_gt_nifti_path = "08_eval_SR_01_HR_GroundTruth.nii.gz" 

# 2. 🌟 模型重建结果
model_sr_nifti_path = "08_eval_SR_03_Reconstructed.nii.gz" 

# ================= 数据预处理工具 =================
def load_and_preprocess(path):
    if not os.path.exists(path):
        print(f"❌ 找不到文件: {path}，请确保文件名拼写正确！")
        sys.exit()
    img = nib.load(path)
    data_vol = img.get_fdata()
    
    # 归一化到 0-255 uint8
    if data_vol.min() < 0:
        data_vol = np.clip(data_vol, -1, 1)
    data_min, data_max = data_vol.min(), data_vol.max()
    if data_max - data_min > 1e-6:
        vol = (data_vol - data_min) / (data_max - data_min) * 255.0
    else:
        vol = data_vol
    return vol.astype(np.uint8), img

# ================= 终极对比核心 =================
def main():
    # 1. 加载完美对齐的 GT
    print(f"📥 正在加载完美对齐的 HR GT: {aligned_gt_nifti_path}...")
    gt_visual, gt_nifti_obj = load_and_preprocess(aligned_gt_nifti_path)

    # 2. 模拟物理插值超分 (Bicubic Baseline)
    print("🎨 正在从对齐的 GT 生成 32^3 -> 64^3 的插值 Baseline...")
    
    # 构建降采样仿射矩阵
    downsample_affine = gt_nifti_obj.affine.copy()
    downsample_affine[:3, :3] *= 2.0
    lr_nifti_obj = nib.Nifti1Image(np.zeros((32, 32, 32)), downsample_affine)
    
    # 降采样到 32 -> 再用 Bicubic 插值放大回 64
    lr_img_obj = resample_from_to(gt_nifti_obj, lr_nifti_obj, order=1) 
    interp_img_obj = resample_from_to(lr_img_obj, gt_nifti_obj, order=3) 
    
    interp_data = interp_img_obj.get_fdata()
    # 同样做归一化
    data_min, data_max = interp_data.min(), interp_data.max()
    interp_visual = ((interp_data - data_min) / (data_max - data_min) * 255.0).astype(np.uint8)

    # 3. 加载模型 SR 结果
    print(f"📥 正在加载模型超分结果: {model_sr_nifti_path}...")
    model_visual, _ = load_and_preprocess(model_sr_nifti_path)

    # 4. 再次安全检查：如果不一致，那真的是见鬼了
    if gt_visual.shape != model_visual.shape:
        print(f"🚨 形状不匹配！GT: {gt_visual.shape}, Model: {model_visual.shape}")
        sys.exit()

    # 5. 计算指标 (见证奇迹的时刻)
    print("\n" + "="*50)
    print("📈 超分辨重建量化对比 (严格空间对齐版):")
    print(f"{'Method':<22} | {'PSNR (↑)':<10} | {'SSIM (↑)':<10} | {'MSE (↓)':<10}")
    print("-" * 55)

    # 插值 Baseline 指标
    psnr_interp = psnr(gt_visual, interp_visual, data_range=255)
    ssim_interp = ssim(gt_visual, interp_visual, data_range=255, win_size=3, channel_axis=None)
    mse_interp = np.mean((gt_visual.astype(float) - interp_visual.astype(float)) ** 2)
    print(f"{'Traditional (Bicubic)':<22} | {psnr_interp:.2f} dB | {ssim_interp:.4f} | {mse_interp:.2f}")

    # 模型指标
    psnr_model = psnr(gt_visual, model_visual, data_range=255)
    ssim_model = ssim(gt_visual, model_visual, data_range=255, win_size=3, channel_axis=None)
    mse_model = np.mean((gt_visual.astype(float) - model_visual.astype(float)) ** 2)
    print(f"{'MAR Model (Ours)':<22} | {psnr_model:.2f} dB | {ssim_model:.4f} | {mse_model:.2f}")
    print("="*50)

    # 6. 生成图表
    vols = [gt_visual, interp_visual, model_visual]
    row_names = ['HR Ground Truth\n(Perfectly Aligned)', 'Traditional Interp.\n(Bicubic SR Baseline)', 'SR Reconstruction\n(Our MAR Model)']
    d_mid, h_mid, w_mid = 32, 32, 32 

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.3) 

    for i in range(3): 
        current_vol = vols[i]
        slices = [current_vol[d_mid, :, :], current_vol[:, h_mid, :], current_vol[:, :, w_mid]]
        
        row_title = row_names[i]
        if i == 1: row_title += f"\n(PSNR: {psnr_interp:.2f} SSIM: {ssim_interp:.3f})"
        elif i == 2: row_title += f"\n(PSNR: {psnr_model:.2f} SSIM: {ssim_model:.3f})"

        for j in range(3): 
            ax = axes[i, j]
            ax.imshow(slices[j], cmap='gray', vmin=0, vmax=255, origin='lower')
            if i == 0: ax.set_title(['Axial', 'Coronal', 'Sagittal'][j], fontsize=14, fontweight='bold')
            if j == 0: ax.set_ylabel(row_title, fontsize=12, fontweight='bold', labelpad=15)
            ax.set_xticks([]); ax.set_yticks([])

    summary_plot_name = "08_final_super_resolution_comparison_aligned.png"
    plt.savefig(summary_plot_name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"🖼️ 严丝合缝版对比图已保存: {summary_plot_name}")

if __name__ == "__main__":
    main()