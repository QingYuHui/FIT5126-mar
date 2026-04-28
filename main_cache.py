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

from util.crop import center_crop_arr

import torch.nn.functional as F

# 3D 模式新增
import nibabel as nib
import glob

# 3D 模式新增
class BraTSDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, img_size=128):
        self.root_dir = root_dir
        self.img_size = img_size
        # 递归查找所有以 t1.nii.gz 结尾的文件
        self.files = sorted(glob.glob(os.path.join(root_dir, "**", "*t1.nii.gz"), recursive=True)) 
        print(f"✅ 找到 {len(self.files)} 个数据文件")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img_nii = nib.load(path)
        img = img_nii.get_fdata()

        # 1. 自动寻找脑组织的边界 (去掉无用的黑边)
        # 寻找所有大于 0 的像素坐标
        mask = img > img.mean() # 简单的阈值分割
        coords = np.argwhere(mask)
        # 找到最小和最大的坐标索引
        y_min, x_min, z_min = coords.min(axis=0)
        y_max, x_max, z_max = coords.max(axis=0)
        
        # 截取有脑子的部分
        img = img[y_min:y_max, x_min:x_max, z_min:z_max]

        # 2. 归一化 (保持不变)
        lower = np.percentile(img, 0.5)
        upper = np.percentile(img, 99.5)
        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower + 1e-8)
        img = img * 2 - 1

        # 3. 缩放与维度对齐
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        
        # 🚀 此时 img 已经是紧贴脑组织的立方体了，Resize 后拉伸感会消失
        img_resized = F.interpolate(
            img_tensor, 
            size=(self.img_size, self.img_size, self.img_size), 
            mode='trilinear', 
            align_corners=False
        ).squeeze(0)

        # 4. 转置为 Axial 视图: [1, H, W, D] -> [1, D, H, W]
        img_resized = img_resized.permute(0, 3, 1, 2)

        filename = os.path.basename(path).replace('.nii.gz', '')
        return img_resized, 0, filename

def get_args_parser():
    parser = argparse.ArgumentParser('Cache VAE latents', add_help=False)
    # 3D 模式建议 Batch Size 设小一点
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')

    # VAE parameters
    # 3D 显存消耗大，128x128x128 比较合适
    parser.add_argument('--img_size', default=128, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="vqgan\stage1.ckpt", type=str,
                        help='images input size')
    parser.add_argument('--vae_embed_dim', default=4, type=int,
                        help='vae output embedding dimension')
    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # caching latents
    parser.add_argument('--cached_path', default='', help='path to cached latents')

    return parser


def main(args):

    # 1. 环境与分布式设置 (Setup)
    misc.init_distributed_mode(args) # 初始化多显卡并行环境

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # 2D 模式
    # 2. 数据加载 (Data Loading)
    # augmentation following DiT and ADM
    # transform_train = transforms.Compose([
    #     transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
    #     # 预处理缓存，我们需要的是图片原本的样子。如果在缓存时随机翻转了，存下来的 Latent 就是翻转过的，这会造成数据的不确定性。
    #     # transforms.RandomHorizontalFlip(), 
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # ])
    # dataset_train = ImageFolderWithFilename(os.path.join(args.data_path, 'train'), transform=transform_train)
    # print(dataset_train)

    # 3D 模式
    # 2. 数据加载 (Data Loading) - 修改为 BraTSDataset
    dataset_train = BraTSDataset(args.data_path, img_size=args.img_size)
    print(dataset_train)

    # <-- 注意！Shuffle 是 False
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False,
    )
    print("Sampler_train = %s" % str(sampler_train))

    # Drop_last=False：哪怕最后一个 Batch 凑不够数，也要处理。一张图都不能少。
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,  # Don't drop in cache
    )

    # 2D 版本
    # 3. 加载 VAE
    # .eval()：开启推理模式（不启用 Dropout 等）。
    # define the vae
    # vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).cuda().eval()

    # 3D 版本 VAE
    # vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), use_variational=False, ckpt_path=args.vae_path).cuda().eval()

    ddconfig = {
        "double_z": False,
        "z_channels": 4,
        "resolution": args.img_size,
        "in_channels": 1,
        "out_ch": 1,
        "ch": 64,
        "num_groups": 32,
        "ch_mult": [1, 1, 2],    # 关键！f4 模型
        "num_res_blocks": 1,     # 关键！
        "attn_resolutions": [],
        "dropout": 0.0
    }
    
    print(f"🚀 初始化 VQGAN (Embed Dim: {args.vae_embed_dim})...")
    vae = AutoencoderKL(ddconfig, 8192, args.vae_embed_dim, args.vae_path).to(device)
    vae = vae.to(device) 
    vae.eval()

    print(f"Start caching VAE latents to {args.cached_path}")
    if args.cached_path:
        os.makedirs(args.cached_path, exist_ok=True)
    
    start_time = time.time()
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Caching: '
    print_freq = 10
    
    # 使用 metric_logger 包装 loader，这样你看日志更舒服
    for data_iter_step, (samples, _, paths) in enumerate(metric_logger.log_every(data_loader_train, print_freq, header)):
        
        # 1. 搬运数据
        samples = samples.to(device, non_blocking=True)
        
        with torch.no_grad():
            # -------------------------------------------------
            # 1. 正常编码 (Original)
            # -------------------------------------------------
            # VQGAN encode 直接返回 Tensor，没有 .parameters 属性
            # 所以我们要把 output 直接赋值给 moments
            posterior = vae.encode(samples)
            
            # 兼容性处理：防止返回 (quant, loss, info)
            if isinstance(posterior, tuple):
                posterior = posterior[0]
                
            moments = posterior # VQGAN 的 "moments" 就是它的 Latent

            # -------------------------------------------------
            # 2. 翻转编码 (Flipping Trick)
            # -------------------------------------------------
            # 3D 版本: (B, C, D, H, W) -> flip dims=[4] (Width方向翻转)
            samples_flip = samples.flip(dims=[4])
            
            posterior_flip = vae.encode(samples_flip)
            
            if isinstance(posterior_flip, tuple):
                posterior_flip = posterior_flip[0]
                
            moments_flip = posterior_flip

        # -------------------------------------------------
        # 3. 存盘 (.npz)
        # -------------------------------------------------
        # 转为 CPU Numpy
        moments_np = moments.cpu().numpy()
        moments_flip_np = moments_flip.cpu().numpy()
        
        for i, path in enumerate(paths):
            # 构造保存路径
            save_path = os.path.join(args.cached_path, path + '.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # ✅ 关键：使用 np.savez 保存两个数组，保持和你旧方案一致的 key
            np.savez(save_path, 
                     moments=moments_np[i], 
                     moments_flip=moments_flip_np[i])

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Caching time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.cached_path:
        Path(args.cached_path).mkdir(parents=True, exist_ok=True)
    main(args)


# -------------------------------------------------------------------
# 把数据集里所有的 RGB 图片，全部跑一遍 VAE 编码器，把得到的 Latent $z$ 保存到硬盘上（通常存为 .npy 文件）。
# 这样做的好处是：在后续训练 MAR 时，不需要每次都重复做“图片 -> Latent”的繁重计算，直接从硬盘读取 Latent 即可，训练速度能快几十倍。
#