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
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.loader import CachedFolder

from models.vae import AutoencoderKL
from models import mar
from engine_mar import train_one_epoch, evaluate
import copy

# 3D 模式新增
import nibabel as nib
import glob


# ================= 修改 1: 定义读取缓存的 Dataset =================
class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # 递归查找 output_cache 下所有的 .npz 文件
        self.files = sorted(glob.glob(os.path.join(root_dir, "**", "*.npz"), recursive=True))
        print(f"✅ 已加载缓存数据集: {self.root_dir}")
        print(f"📊 样本数量: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            data = np.load(path)
            # 读取 Latent (moments)
            # 形状应该是 [4, 32, 32, 32] (对应 128 的输入)
            z = data['moments'] 
            
            # 训练 MAR 需要 label，BraTS 没有类别，固定返回 0
            return torch.from_numpy(z).float(), 0 
        except Exception as e:
            print(f"❌ 读取缓存失败 {path}: {e}")
            # 返回全 0 防止训练中断
            return torch.zeros((4, 32, 32, 32)), 0


def get_args_parser():
    # === 修改: 3D 显存压力大，建议 batch_size 减小到 4 ===
    parser = argparse.ArgumentParser('MAR training with Diffusion Loss', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=400, type=int)

    # Model parameters
    parser.add_argument('--model', default='mar_large', type=str, metavar='MODEL',
                        help='Name of model to train')

    # VAE parameters
    # === 修改: 3D 默认 img_size 建议 64 或 128 ===
    parser.add_argument('--img_size', default=128, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='images input size')
    parser.add_argument('--vae_embed_dim', default=4, type=int,
                        help='vae output embedding dimension')
    parser.add_argument('--vae_stride', default=4, type=int,
                        help='tokenizer stride, default use KL16')
    parser.add_argument('--patch_size', default=1, type=int,
                        help='number of tokens to group as a patch.')

    # Generation parameters
    parser.add_argument('--num_iter', default=64, type=int,
                        help='number of autoregressive iterations to generate an image')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='number of images to generate')
    parser.add_argument('--cfg', default=1.0, type=float, help="classifier-free guidance")
    parser.add_argument('--cfg_schedule', default="linear", type=str)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--eval_freq', type=int, default=40, help='evaluation frequency')
    parser.add_argument('--save_last_freq', type=int, default=5, help='save last frequency')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate', action='store_true')

    # === 修改: 3D 评估时的 Batch Size 也相应减小, 例如 4 ===
    parser.add_argument('--eval_bsz', type=int, default=64, help='generation batch size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')

    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--ema_rate', default=0.9999, type=float)

    # MAR params
    parser.add_argument('--mask_ratio_min', type=float, default=0.7,
                        help='Minimum mask ratio')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='projection dropout')
    parser.add_argument('--buffer_size', type=int, default=64)

    # Diffusion Loss params
    parser.add_argument('--diffloss_d', type=int, default=12)
    parser.add_argument('--diffloss_w', type=int, default=1536)
    parser.add_argument('--num_sampling_steps', type=str, default="100")
    parser.add_argument('--diffusion_batch_mul', type=int, default=1)
    parser.add_argument('--temperature', default=1.0, type=float, help='diffusion loss sampling temperature')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    
    # === 修改: BraTS 默认为 1 类 ===
    parser.add_argument('--class_num', default=1, type=int)
    # 2D 模式
    # parser.add_argument('--class_num', default=1000, type=int)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
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
    parser.add_argument('--use_cached', action='store_true', dest='use_cached',
                        help='Use cached latents')
    parser.set_defaults(use_cached=True)
    parser.add_argument('--cached_path', default='./output_cache', type=str)

    return parser


def main(args):
    # 检测环境变量，初始化分布式训练
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # 设定随机种子 (Seed)
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # num_tasks (World Size)：总共有多少个进程
    # global_rank (Rank)：当前进程的 ID
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # ------ 预处理步骤 ------
    # center_crop_arr: 把图片中心裁剪成 256x256(例如)
    # RandomHorizontalFlip: 随机水平翻转（增加数据多样性）
    # Normalize: 把像素值归一化到 -1 到 1 之间

    # 2D 模式
    # augmentation following DiT and ADM
    # 1. 定义数据增强：中心裁剪、翻转、归一化
    # transform_train = transforms.Compose([
    #     transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # ])

    # 2. 加载数据集：通常是 ImageFolder 格式
    dataset_train = CachedDataset(args.cached_path)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # 🔥 探针: 检查数据形状 🔥
    try:
        sample_img, _ = dataset_train[0]
        print(f"👉 Dataset Output Shape: {sample_img.shape}") 
        # 应该输出 [4, 32, 32, 32]
    except Exception as e:
        print(f"❌ Dataset Check Failed: {e}")

    # =========================================================
    # 2. 初始化 VAE (仅用于 Evaluation 看图，不参与训练)
    # =========================================================

    ddconfig_verified = {
        "double_z": False,       # VQGAN 关键
        "z_channels": 4,         # Latent 维度
        "resolution": 64,        # 输入大小 (不严格限制)
        "in_channels": 1,        # 灰度
        "out_ch": 1,             # 灰度输出
        "ch": 64,                # 基础通道
        "num_groups": 32,
        "ch_mult": [1, 1, 2],    # 【关键验证结果】3层结构
        "num_res_blocks": 1,     # 【关键验证结果】每层1个残差块
        "attn_resolutions": [],
        "dropout": 0.0
    }

    print("正在初始化修改后的 VQModel3D...")
    print("🚀 初始化 VAE (用于评估解码)...")
    # 注意变量名是 vae，不要用 model
    vae = AutoencoderKL(ddconfig_verified, 8192, 4, args.vae_path).to(device)
    print("✅ 模型结构初始化成功！现在它是一个真正的 VQGAN 了。")

    # 关键点: 冻结参数
    vae.to(device).eval()
    for param in vae.parameters():
        param.requires_grad = False

    # ------ 构建“大脑” MAR (Build MAR Model) ------
    model = mar.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        mask_ratio_min=args.mask_ratio_min,
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        grad_checkpointing=args.grad_checkpointing,
    )

    print("Model = %s" % str(model))
    # following timm: set wd as 0 for bias and norm layers
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))

    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * misc.get_world_size()

    # 学习率 (lr) 计算
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # ------ 设置优化器 ------
    # 1. 权重衰减 (Weight Decay) 设置
    # 排除了 Bias（偏置）和 Norm（归一化）层，不对它们做权重衰减
    # no weight decay on bias, norm layers, and diffloss MLP
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    # 2. AdamW 优化器：负责更新参数的大脑
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # EMA 全称是 Exponential Moving Average,
    # “指数移动平均”。它不仅保存当前训练的参数，还保存一份“历史平均”参数。通常 EMA 的模型在生成图片时效果更平滑、更好
    
    # resume training
    # ------ 断点续训 ------
    if args.resume and os.path.exists(os.path.join(args.resume, "checkpoint-last.pth")):
        checkpoint = torch.load(os.path.join(args.resume, "checkpoint-last.pth"), map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        model_params = list(model_without_ddp.parameters())
        ema_state_dict = checkpoint['model_ema']
        ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
        del checkpoint
    else:
        model_params = list(model_without_ddp.parameters())
        ema_params = copy.deepcopy(model_params)
        print("Training from scratch")

    # evaluate FID and IS
    if args.evaluate:
        torch.cuda.empty_cache()
        evaluate(model_without_ddp, vae, ema_params, args, 0, batch_size=args.eval_bsz, log_writer=log_writer,
                 cfg=args.cfg, use_ema=True)
        return

    # training
    # ------ 正式训练循环 ------
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # 1.【核心训练步骤】
        # 输入：从 DataLoader 拿一批 RGB 图片
        # VAE 压缩：把 RGB 图片丢进锁死的 VAE，得到 Latent x
        # MAR 前向：随机 Mask 掉 x 的一部分, 把 Latent x 丢进 MAR，计算 Diffusion Loss
        # 反向传播：Optimizer 根据 Loss 修改 MAR 的参数
        train_one_epoch(
            model, vae,
            model_params, ema_params,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # 2. 保存模型 (Checkpointing)
        # 每隔几轮（save_last_freq），就把当前的模型权重保存下来
        # save checkpoint
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, epoch_name="last")

        # 3. 在线评估 (Online Evaluation)
        # 每训练 eval_freq 轮，暂停训练
        # 用当前的 MAR 模型生成一批图片, 计算 FID 分数（一种衡量生成图片真实度和多样性的指标）
        # 如果不达标，就继续训练
        # online evaluation
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=args.eval_bsz, log_writer=log_writer,
                     cfg=1.0, use_ema=True)
            if not (args.cfg == 1.0 or args.cfg == 0.0):
                evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=args.eval_bsz // 2,
                         log_writer=log_writer, cfg=args.cfg, use_ema=True)
            torch.cuda.empty_cache()

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)


# -------------------------------------------------------------------
# 假设配置如下：
# Batch Size (B) = 1
# Image Size = 256x256
# VAE Latent Dim = 16 (AutoencoderKL 的输出通道)
# Patch Size = 1
# Transformer Embed Dim = 1024 (MAR 内部特征维度)
# Sequence Length (L) = 256 (16x16)
#
# 第一阶段：训练数据流 (Training Data Flow)
# 训练的目的是：优化 MAR (Transformer) 和 SimpleMLPAdaLN (Diffusion MLP) 的参数。
# 1. 输入预处理 (VAE Context)
# 输入: 原始图像 $x_{rgb}$，形状 [1, 3, 256, 256]。
# 处理: 调用 AutoencoderKL.encode(x_{rgb})。
# 输出: 潜在特征图 (Latent Map)，形状 [1, 16, 16, 16]。
# 注：在 MAR 类的 forward 中，输入的 imgs 通常已经是这个 Latent Map。
#
# 2. 序列化与掩码 (Inside MAR)
# Patchify:
# 调用 MAR.patchify(imgs)。
# 将 [1, 16, 16, 16] 变形为 [1, 256, 16]。
# 变量名: gt_latents (Ground Truth Latents)。这是 Diffusion 的预测目标 (Target)。
# Random Masking:
# 生成掩码 mask，形状 [1, 256] (假设 75% 为 1，即被遮挡)。
# 保留可见的 Token，形状 [1, 64, 16] (256 * 0.25)。
#
# 3. 上下文编码 (MAR Transformer)
#
# Encoder (forward_mae_encoder):
# 输入: 可见 Token [1, 64, 16] + class_embedding + 位置编码。
# 映射: Linear 层将 16 维映射到 1024 维。
# 处理: 经过 Vision Transformer Blocks。
#输出: 编码特征 [1, 65, 1024] (含 Buffer/Class Token)。

# Decoder (forward_mae_decoder):
# 输入: Encoder 输出 + mask_token (填充被 Mask 的 192 个位置)。完整序列长度恢复为 256。
# 处理: 经过 Vision Transformer Blocks。
# 输出: 变量名为 z，形状 [1, 256, 1024]。
# 关键定义: 这个 z 是 Diffusion 的条件 (Conditioning)，它包含了模型对每个位置的“语义理解”。
#
# 4. 扩散损失计算 (Inside DiffLoss)
# 调用: MAR 调用 self.diffloss(z=z, target=gt_latents, mask=mask)。
#
# 输入对齐:
# target (真值): [1, 256, 16] (来自 VAE)。
# z (条件): [1, 256, 1024] (来自 MAR Transformer)。
# 
# 扩散过程 (DiffLoss.forward):
# 采样时间: 随机采样时间步 t (例如 $t=500$)。
# 加噪: 对 target 添加高斯噪声 $\epsilon$，得到 $x_t$。
# 预测: 调用 self.net (即 SimpleMLPAdaLN)。
#       self.net(x_t, t, c=z)。
#       AdaLN 机制: SimpleMLPAdaLN 内部使用 z 生成 shift/scale 参数，调制 ResBlock 的特征图，指导去噪。
# 计算 Loss: 计算预测噪声 $\hat{\epsilon}$ 与真实噪声 $\epsilon$ 之间的 MSE Loss。
# Mask 过滤: 只计算 mask=1 (被遮挡) 位置的 Loss，忽略可见部分。
#
# -------------------------------------------------------------------
# 第二阶段：推理/生成数据流 (Inference/Sampling Data Flow)
# 推理的目的是：从全白 Mask 和 Label 开始，生成完整的 Latent Map。
# 
# 1. 初始化
# tokens: 全 0 张量 [1, 256, 16]。
# mask: 全 1 张量 [1, 256] (表示全未知)。
# labels: 目标类别 (例如 "Dog")。
# 
# 2. 迭代生成循环 (Loop inside sample_tokens)假设共 16 步，当前是第 $i$ 步：
# 步骤 A: 生成条件 (MAR Transformer)
# 输入: 当前的 tokens (部分已知，部分未知) 和 mask。
# 处理: 运行 forward_mae_encoder 和 forward_mae_decoder。
# 输出: 条件张量 z，形状 [1, 256, 1024]。
# 
# 步骤 B: 扩散采样 (DiffLoss & SimpleMLPAdaLN)
# 准备 CFG: 将 z 复制拼接：cat([z_cond, z_uncond])，形状变为 [2, 256, 1024]。
# 调用: self.diffloss.sample(z, cfg=scale)。
# 去噪循环 (p_sample_loop):
# 初始化纯噪声 noise [1, 256, 16]。
# 从 $T=1000$ 到 $0$ 循环调用 self.net.forward_with_cfg(x, t, c=z)。

# SimpleMLPAdaLN 内部:
# 分别计算有条件输出 $\epsilon_{cond}$ (基于 "Dog") 和无条件输出 $\epsilon_{uncond}$ (基于 fake_latent)。
# 应用公式: $\epsilon_{final} = \epsilon_{uncond} + s \cdot (\epsilon_{cond} - \epsilon_{uncond})$。
# 
# 输出: sampled_token_latent，形状 [1, 256, 16]。这是扩散模型认为“这一轮完整的图应该是这个样子”。
# 
# 步骤 C: 更新画布 (Update Tokens)
# 计算当前步需要确定的位置 mask_to_pred (根据 MaskGIT 调度策略)。
# 从 sampled_token_latent 中取出对应位置的值，填入 tokens。
# 更新 mask (将已填入位置置为 0)。
# 
# 3. 最终还原 (Post-Processing)
# Unpatchify: MAR.unpatchify(tokens) $\to$ [1, 16, 16, 16]。
# Decode: AutoencoderKL.decode(latents) $\to$ [1, 3, 256, 256] (RGB 图像)。
##