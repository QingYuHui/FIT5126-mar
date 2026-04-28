import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import util.misc as misc
from util.loader import ImageFolderWithFilename

from models.vae import AutoencoderKL
from engine_mar import cache_latents

import torch.nn.functional as F
import nibabel as nib
import glob

# ==========================================
# 1. 自适应的 BraTSDataset (支持 64 和 128)
# ==========================================
class BraTSDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, img_size=128):
        self.root_dir = root_dir
        self.img_size = img_size
        self.files = sorted(glob.glob(os.path.join(root_dir, "**", "*t1.nii.gz"), recursive=True)) 
        print(f"✅ 找到 {len(self.files)} 个数据文件，当前模式尺寸: {self.img_size}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img_nii = nib.load(path)
        img = img_nii.get_fdata()

        # 1. 自动寻找脑组织边界 (去黑边)
        mask = img > img.mean()
        coords = np.argwhere(mask)
        if len(coords) > 0:
            y_min, x_min, z_min = coords.min(axis=0)
            y_max, x_max, z_max = coords.max(axis=0)
            img = img[y_min:y_max, x_min:x_max, z_min:z_max]

        # 2. 归一化 [-1, 1]
        lower = np.percentile(img, 0.5)
        upper = np.percentile(img, 99.5)
        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower + 1e-8)
        img = img * 2 - 1

        filename = os.path.basename(path).replace('.nii.gz', '')

        # ----------------------------------------
        # 策略 A: 64 尺寸 -> 物理局部随机裁剪
        # ----------------------------------------
        if self.img_size == 64:
            d_img, h_img, w_img = img.shape
            crop_d, crop_h, crop_w = min(64, d_img), min(64, h_img), min(64, w_img)

            z_start = np.random.randint(0, d_img - crop_d + 1) if d_img > crop_d else 0
            y_start = np.random.randint(0, h_img - crop_h + 1) if h_img > crop_h else 0
            x_start = np.random.randint(0, w_img - crop_w + 1) if w_img > crop_w else 0

            img_crop = img[z_start:z_start+crop_d, y_start:y_start+crop_h, x_start:x_start+crop_w]

            # 尺寸不够时用 -1 (纯黑) 填充保护
            if img_crop.shape != (64, 64, 64):
                pad_d = 64 - img_crop.shape[0]
                pad_h = 64 - img_crop.shape[1]
                pad_w = 64 - img_crop.shape[2]
                img_crop = np.pad(img_crop, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=-1.0)

            img_tensor = torch.from_numpy(img_crop).float().unsqueeze(0) # [1, D, H, W]
            return img_tensor, 0, filename

        # ----------------------------------------
        # 策略 B: 128 尺寸 -> Resize 后切成 8 块
        # ----------------------------------------
        elif self.img_size == 128:
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
            img_128 = F.interpolate(img_tensor, size=(128, 128, 128), mode='trilinear', align_corners=False)
            img_128 = img_128.squeeze(0).squeeze(0).numpy()

            blocks = []
            for z in [0, 64]:
                for y in [0, 64]:
                    for x in [0, 64]:
                        block = img_128[z:z+64, y:y+64, x:x+64]
                        block_tensor = torch.from_numpy(block).float().unsqueeze(0) # [1, D, H, W]
                        blocks.append(block_tensor)
            
            blocks_tensor = torch.stack(blocks, dim=0) # [8, 1, 64, 64, 64]
            return blocks_tensor, 0, filename
        else:
            raise ValueError("不支持的 img_size，请使用 64 或 128")


def get_args_parser():
    parser = argparse.ArgumentParser('Cache VAE latents', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--img_size', default=128, type=int, help='images input size (64 or 128)')
    parser.add_argument('--vae_path', default="vqgan/stage1.ckpt", type=str)
    parser.add_argument('--vae_embed_dim', default=4, type=int)
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--cached_path', default='', help='path to cached latents')
    return parser


def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    dataset_train = BraTSDataset(args.data_path, img_size=args.img_size)
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    # VAE 的输入块固定是 64，所以 resolution 写死 64 最安全
    ddconfig = {
        "double_z": False, "z_channels": 4, "resolution": 64, 
        "in_channels": 1, "out_ch": 1, "ch": 64, 
        "num_groups": 32, "ch_mult": [1, 1, 2], 
        "num_res_blocks": 1, "attn_resolutions": [], "dropout": 0.0
    }
    
    print(f"🚀 初始化 VQGAN...")
    vae = AutoencoderKL(ddconfig, 8192, args.vae_embed_dim, args.vae_path).to(device)
    vae.eval()

    if args.cached_path:
        os.makedirs(args.cached_path, exist_ok=True)
    
    start_time = time.time()
    metric_logger = misc.MetricLogger(delimiter="  ")
    
    # ==========================================
    # 2. 自适应保存逻辑 (处理 64的单块 vs 128的多块)
    # ==========================================
    for data_iter_step, (samples, _, paths) in enumerate(metric_logger.log_every(data_loader_train, 10, 'Caching: ')):
        
        # 判断当前批次的数据是不是 "一拆八" 模式 (维度是 6 维: [B, 8, C, D, H, W])
        is_split_mode = (samples.dim() == 6)
        if is_split_mode:
            B, num_blocks, C, D, H, W = samples.shape
            # 把 8 个块平铺到 Batch 维度上，变成 [B*8, C, D, H, W]，让 VAE 一次性处理
            samples = samples.view(-1, C, D, H, W)
            
        samples = samples.to(device, non_blocking=True)
        
        with torch.no_grad():
            posterior = vae.encode(samples)
            if isinstance(posterior, tuple): posterior = posterior[0]
            moments = posterior

            samples_flip = samples.flip(dims=[4])
            posterior_flip = vae.encode(samples_flip)
            if isinstance(posterior_flip, tuple): posterior_flip = posterior_flip[0]
            moments_flip = posterior_flip

        moments_np = moments.cpu().numpy()
        moments_flip_np = moments_flip.cpu().numpy()
        
        # 存盘逻辑
        if is_split_mode:
            # 128 模式：把铺平的块重新对应回原来的文件名，加上 _part0 到 _part7
            for b_idx, path in enumerate(paths):
                for block_idx in range(num_blocks):
                    global_idx = b_idx * num_blocks + block_idx
                    save_path = os.path.join(args.cached_path, f"{path}_part{block_idx}.npz")
                    np.savez(save_path, 
                             moments=moments_np[global_idx], 
                             moments_flip=moments_flip_np[global_idx])
        else:
            # 64 模式：正常保存
            for i, path in enumerate(paths):
                save_path = os.path.join(args.cached_path, path + '.npz')
                np.savez(save_path, 
                         moments=moments_np[i], 
                         moments_flip=moments_flip_np[i])

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f'Caching time {total_time_str}')

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.cached_path:
        Path(args.cached_path).mkdir(parents=True, exist_ok=True)
    main(args)