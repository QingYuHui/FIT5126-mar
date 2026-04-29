"""
Purpose:
    Controlled latent-space super-resolution-style evaluation. The script
    degrades a high-resolution cached latent volume by masking structured
    latent anchors, reconstructs missing tokens with MAR, decodes the result,
    and saves qualitative outputs for the thesis results section.

Suggested filename:
    22_eval_latent_space_super_resolution.py

Notes:
    This is an SR-style latent reconstruction test, not a fully conditional
    LR-MRI to HR-MRI clinical super-resolution pipeline.
"""

# Allow this copied script to be run from either the repository root or Evaluate/.
from pathlib import Path
import sys
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

OUTPUT_DIR = REPO_ROOT / "Evaluate" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def out_path(filename: str) -> str:
    return str(OUTPUT_DIR / filename)



import math
import os
import random

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models import mar, vae


# ---------------------------------------------------------------------
# Controlled latent-space SR-style evaluation.
#
# This is not yet a fully conditional LR-MRI -> HR-MRI model. It evaluates
# whether the trained MAR prior can fill missing high-resolution latent tokens
# from a sparse/degraded latent observation.
# ---------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAE_PATH = "vqgan/stage1.ckpt"
CHECKPOINT_PATH = "output_run_64_patch1/checkpoint-last.pth"
TARGET_NPZ = "output_cache/class0/BraTS2021_00002_t1.npz"

SCALE_FACTOR = 2.6
NUM_ITER = 128
TEMPERATURE = 0.80
CFG = 1.0
SEED = 11

# "plane_grid" keeps 1/4 latent anchors and is much more stable for figures.
# "strict_3d_grid" keeps 1/8 anchors and is a harder stress test.
# "random" keeps a configurable random fraction of tokens.
SR_MASK_MODE = "plane_grid"
RANDOM_KEEP_RATIO = 0.50


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_visual_uint8(vol: np.ndarray) -> np.ndarray:
    vol = np.clip(vol, -1, 1)
    return ((vol + 1.0) / 2.0 * 255.0).astype(np.uint8)


def save_viewable_nifti(img_data: np.ndarray, filename: str) -> None:
    img = to_visual_uint8(img_data)
    nib.save(nib.Nifti1Image(img, np.eye(4)), filename)
    print(f"Saved NIfTI: {filename}")


def load_vae() -> torch.nn.Module:
    ddconfig = {
        "double_z": False,
        "z_channels": 4,
        "resolution": 64,
        "in_channels": 1,
        "out_ch": 1,
        "ch": 64,
        "num_groups": 32,
        "ch_mult": [1, 1, 2],
        "num_res_blocks": 1,
        "attn_resolutions": [],
        "dropout": 0.0,
    }
    vae_model = vae.AutoencoderKL(ddconfig, 8192, 4)
    if not os.path.exists(VAE_PATH):
        raise FileNotFoundError(f"VAE checkpoint not found: {VAE_PATH}")

    sd = torch.load(VAE_PATH, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    try:
        vae_model.load_state_dict(sd, strict=True)
        print("VAE weights loaded with strict=True.")
    except RuntimeError as exc:
        print("Warning: VAE strict load failed; falling back to strict=False.")
        print(exc)
        vae_model.load_state_dict(sd, strict=False)
    return vae_model.to(DEVICE).eval()


def load_mar() -> torch.nn.Module:
    model = mar.mar_base(
        img_size=64,
        vae_stride=4,
        vae_embed_dim=4,
        patch_size=1,
        num_sampling_steps="100",
        diffloss_d=12,
        diffloss_w=1536,
    ).to(DEVICE)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"MAR checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = checkpoint.get("model_ema", checkpoint.get("model", checkpoint))
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print("MAR weights loaded.")
    return model.eval()


def decode_latent(vae_model: torch.nn.Module, z_latent: torch.Tensor) -> np.ndarray:
    z = z_latent / SCALE_FACTOR
    if hasattr(vae_model, "post_quant_conv"):
        z = vae_model.post_quant_conv(z)
    recon = vae_model.decoder(z)
    return recon[0, 0].detach().cpu().numpy()


def reference_clamp_range(gt_tokens: torch.Tensor) -> tuple[float, float]:
    flat = gt_tokens.detach().flatten()
    lo = torch.quantile(flat, 0.001).item()
    hi = torch.quantile(flat, 0.999).item()
    margin = 0.10 * max(hi - lo, 1e-6)
    return lo - margin, hi + margin


def make_sr_mask(gt_tokens: torch.Tensor, mode: str = SR_MASK_MODE) -> torch.Tensor:
    bsz, seq_len, _ = gt_tokens.shape
    grid_size = int(round(seq_len ** (1 / 3)))
    if grid_size ** 3 != seq_len:
        raise ValueError(f"Expected cubic latent grid, got seq_len={seq_len}")

    if mode == "strict_3d_grid":
        mask_3d = torch.ones((grid_size, grid_size, grid_size), device=DEVICE)
        mask_3d[::2, ::2, ::2] = 0.0  # keep 1/8 anchors
        fixed_mask = mask_3d.reshape(1, seq_len).expand(bsz, -1)
    elif mode == "plane_grid":
        mask_3d = torch.ones((grid_size, grid_size, grid_size), device=DEVICE)
        mask_3d[::2, ::2, :] = 0.0  # keep 1/4 anchors, easier and more stable
        fixed_mask = mask_3d.reshape(1, seq_len).expand(bsz, -1)
    elif mode == "random":
        keep_ratio = float(RANDOM_KEEP_RATIO)
        if not 0.0 < keep_ratio < 1.0:
            raise ValueError("RANDOM_KEEP_RATIO must be in (0, 1)")
        noise = torch.rand(bsz, seq_len, device=DEVICE)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        len_keep = int(seq_len * keep_ratio)
        mask_tokens = torch.ones((bsz, seq_len), device=DEVICE)
        mask_tokens[:, :len_keep] = 0.0
        fixed_mask = torch.gather(mask_tokens, dim=1, index=ids_restore)
    else:
        raise ValueError(f"Unknown SR_MASK_MODE: {mode}")

    print(f"SR mask mode: {mode}; missing latent tokens: {fixed_mask[0].mean().item():.2%}")
    return fixed_mask


def super_resolution_sampling(
    model: torch.nn.Module,
    gt_tokens: torch.Tensor,
    fixed_mask: torch.Tensor,
    num_iter: int = NUM_ITER,
    temperature: float = TEMPERATURE,
    cfg: float = CFG,
) -> torch.Tensor:
    bsz, seq_len, _ = gt_tokens.shape
    tokens = gt_tokens.clone()
    tokens[fixed_mask.bool()] = 0
    mask = fixed_mask.clone()
    clamp_min, clamp_max = reference_clamp_range(gt_tokens)

    num_unknowns = int(fixed_mask[0].sum().item())
    print(f"Latent SR reconstruction: reconstructing {num_unknowns / seq_len:.2%} tokens.")

    for step in range(num_iter):
        cur_tokens = tokens.clone()

        labels = torch.zeros(bsz, dtype=torch.long, device=DEVICE)
        class_embedding = model.class_emb(labels)

        if cfg != 1.0:
            tokens_in = torch.cat([tokens, tokens], dim=0)
            class_embedding = torch.cat([class_embedding, model.fake_latent.repeat(bsz, 1)], dim=0)
            mask_in = torch.cat([mask, mask], dim=0)
        else:
            tokens_in, mask_in = tokens, mask

        x = model.forward_mae_encoder(tokens_in, mask_in, class_embedding)
        z = model.forward_mae_decoder(x, mask_in)

        mask_ratio_step = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
        current_mask_len = int(np.floor(num_unknowns * mask_ratio_step))

        mask_next = torch.zeros_like(mask)
        if current_mask_len > 0:
            random_scores = torch.rand((bsz, seq_len), device=DEVICE)
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
        sampled_token_latent = torch.clamp(sampled_token_latent, clamp_min, clamp_max)

        if cfg != 1.0:
            sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
            mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

        cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
        cur_tokens[~fixed_mask.bool()] = gt_tokens[~fixed_mask.bool()]
        tokens = cur_tokens.clone()

    return model.unpatchify(tokens)


def choose_representative_slices(vol_uint8: np.ndarray) -> tuple[int, int, int]:
    threshold = max(10, int(np.percentile(vol_uint8, 70)))
    fg = vol_uint8 > threshold
    if fg.sum() == 0:
        return tuple(s // 2 for s in vol_uint8.shape)
    d_idx = int(np.argmax(fg.sum(axis=(1, 2))))
    h_idx = int(np.argmax(fg.sum(axis=(0, 2))))
    w_idx = int(np.argmax(fg.sum(axis=(0, 1))))
    return d_idx, h_idx, w_idx


def save_summary_plot(gt_vol: np.ndarray, degraded_vol: np.ndarray, pred_vol: np.ndarray, filename: str) -> None:
    vols = [to_visual_uint8(gt_vol), to_visual_uint8(degraded_vol), to_visual_uint8(pred_vol)]
    row_names = [
        "HR reference\n(VQGAN decoded)",
        f"Degraded latent input\n({SR_MASK_MODE})",
        "MAR latent SR reconstruction",
    ]
    d_idx, h_idx, w_idx = choose_representative_slices(vols[0])

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.08, hspace=0.30)

    for i, current_vol in enumerate(vols):
        slices = [
            current_vol[d_idx, :, :],
            current_vol[:, h_idx, :],
            current_vol[:, :, w_idx],
        ]
        for j, slc in enumerate(slices):
            ax = axes[i, j]
            ax.imshow(slc, cmap="gray", vmin=0, vmax=255, origin="lower")
            if i == 0:
                ax.set_title(["Axial", "Coronal", "Sagittal"][j], fontsize=14, fontweight="bold")
            if j == 0:
                ax.set_ylabel(row_names[i], fontsize=11, fontweight="bold", labelpad=12)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SR summary plot: {filename}")


def main() -> None:
    set_seed(SEED)
    vae_model = load_vae()
    model = load_mar()

    if not os.path.exists(TARGET_NPZ):
        raise FileNotFoundError(f"Cached latent file not found: {TARGET_NPZ}")

    data = np.load(TARGET_NPZ)
    z_gt = torch.from_numpy(data["moments"]).float().to(DEVICE).unsqueeze(0)
    z_gt = z_gt * SCALE_FACTOR

    gt_tokens = model.patchify(z_gt)
    fixed_mask = make_sr_mask(gt_tokens)

    with torch.no_grad():
        reconstructed_z = super_resolution_sampling(model, gt_tokens, fixed_mask)

        degraded_tokens = gt_tokens.clone()
        degraded_tokens[fixed_mask.bool()] = 0
        degraded_input_z = model.unpatchify(degraded_tokens)

        img_gt = decode_latent(vae_model, z_gt)
        img_degraded = decode_latent(vae_model, degraded_input_z)
        img_pred = decode_latent(vae_model, reconstructed_z)

    save_viewable_nifti(img_gt, out_path("22_sr_01_hr_ground_truth.nii.gz"))
    save_viewable_nifti(img_degraded, out_path("22_sr_02_degraded_latent_input.nii.gz"))
    save_viewable_nifti(img_pred, out_path("22_sr_03_mar_reconstruction.nii.gz"))

    gt_u8 = to_visual_uint8(img_gt)
    pred_u8 = to_visual_uint8(img_pred)
    mse = np.mean((gt_u8.astype(np.float32) - pred_u8.astype(np.float32)) ** 2)
    val_psnr = psnr(gt_u8, pred_u8, data_range=255)
    val_ssim = ssim(gt_u8, pred_u8, data_range=255, win_size=3, channel_axis=None)

    save_summary_plot(img_gt, img_degraded, img_pred, out_path("22_sr_summary_plot.png"))

    print("\nLatent-space SR-style reconstruction metrics:")
    print(f"  Mask mode: {SR_MASK_MODE}")
    print(f"  MSE : {mse:.2f}")
    print(f"  PSNR: {val_psnr:.2f} dB")
    print(f"  SSIM: {val_ssim:.4f}")


if __name__ == "__main__":
    main()
