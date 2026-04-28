import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from models.vae import DiagonalGaussianDistribution
import torch_fidelity
import shutil
import cv2
import numpy as np
import os
import copy
import time

# 3D 新增
import nibabel as nib

# 权重的“平滑移动”
# EMA (Exponential Moving Average)：指数移动平均。
# 作用：在训练深度生成模型（GAN, Diffusion）时，模型的参数波动很大。如果我们直接用当前这一步的参数去生成图片，效果可能不稳定。
# 做法：我们维护一套“影子参数”（EMA Params）。
# EMA参数 = 旧EMA参数 * 0.99 + 新模型参数 * 0.01
# 结果：EMA 参数是过去一段时间参数的“平均值”，它更平滑、更鲁棒。通常在测试和生成图片时，我们用 EMA 参数，而不是当前的实时参数。
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


# 训练一轮的完整流程
def train_one_epoch(model, vae,
                    model_params, ema_params,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # ==============================================
        # 🔥🔥 探针 2: 检查进入 GPU 前的 Batch 形状 🔥🔥
        # ==============================================
        # if data_iter_step == 0: # 只打印第一个 Batch，防止刷屏
        #     print(f"\n[Engine Debug] Batch Input Shape: {samples.shape}")
        #     if samples.shape[1] == 2:
        #         print("🚨 警报: 进入训练循环时已经是 2 通道了！")
        # ==============================================

        # 调整学习率：lr_sched.adjust_learning_rate。注意它是 per iteration（每一步都调），而不是每轮调一次，这样更细腻。
        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            # 情况 A (use_cached=True)：samples 已经是读进来的 .npy 数据（均值和方差）。直接构建分布。
            # 情况 B (use_cached=False)：samples 是 RGB 图片。需要丢进 VAE 编码器，得到均值和方差。
            # if args.use_cached:
            #     moments = samples
            #     posterior = DiagonalGaussianDistribution(moments)
            # else:
            #     posterior = vae.encode(samples)

            # 3D 版本
            # 缓存已经是 4 通道了，直接把它当作 latent x
            x = samples

            # 特殊处理, 乘一个系数放大 latent 的数值
            scale_factor = 2.6 
            x = x * scale_factor

            # normalize the std of latent to be 1. Change it if you use a different tokenizer
            # 0.2325 是一个魔法数字（Magic Number）
            # 来自 Stable Diffusion 的论文。因为 VAE 压缩后的 latent 方差很大，
            # 为了让它接近标准正态分布 (std=1)，需要乘一个系数缩放一下。
            # 3D 模式下关了
            # x = posterior.sample().mul_(0.2325)

        # print(f"[Engine Final Check] Shape feeding to model: {samples.shape}")

        # forward
        # 前向传播
        # 3D MAR 前向传播
        with torch.cuda.amp.autocast(): # 开启混合精度训练（自动把 float32 转 float16），省显存、跑得快。
            # 调用 MAR 模型，内部会跑 Transformer 和 DiffLoss。
            loss = model(x, labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # 反向传播与优化
        # 处理梯度缩放（配合混合精度），防止梯度下溢。
        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        # 更新 EMA：每次更新完主模型，顺手把 EMA 模型也挪动一点点。
        update_ema(ema_params, model_params, rate=args.ema_rate)

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# 评估, 每训练几十轮，就要停下来看看模型学得怎么样。
def evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=16, log_writer=None, cfg=1.0,
             use_ema=True):
    model_without_ddp.eval()
    num_steps = args.num_images // (batch_size * misc.get_world_size()) + 1
    save_folder = os.path.join(args.output_dir, "ariter{}-diffsteps{}-temp{}-{}cfg{}-image{}".format(args.num_iter,
                                                                                                     args.num_sampling_steps,
                                                                                                     args.temperature,
                                                                                                     args.cfg_schedule,
                                                                                                     cfg,
                                                                                                     args.num_images))
    
    # 先把当前练到一半的模型参数存起来 (model_state_dict)。
    # 把 EMA 参数加载进模型（因为 EMA 生成效果更好）。
    if use_ema:
        save_folder = save_folder + "_ema"
    if args.evaluate:
        save_folder = save_folder + "_evaluate"
    print("Save to:", save_folder)
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    # switch to ema params
    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    class_num = args.class_num
    if args.num_images < class_num:
        # 比如你生成4张，这里就只取类别 0, 1, 2, 3
        class_label_gen_world = np.arange(0, args.num_images)
    else:
        # 原有逻辑
        class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)

    # class_num = args.class_num
    # assert args.num_images % class_num == 0  # number of images per class must be the same
    # class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num) # 构造一批标签（比如生成 50 张狗，50 张猫...）。

    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    used_time = 0
    gen_img_cnt = 0

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        labels_gen = class_label_gen_world[world_size * batch_size * i + local_rank * batch_size:
                                                world_size * batch_size * i + (local_rank + 1) * batch_size]
        labels_gen = torch.Tensor(labels_gen).long().cuda()


        torch.cuda.synchronize()
        start_time = time.time()

        # generation
        # MAR 逐步生成 Latent。
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # 3D MAR 生成
                sampled_tokens = model_without_ddp.sample_tokens(bsz=batch_size, num_iter=args.num_iter, cfg=cfg,
                                                                 cfg_schedule=args.cfg_schedule, labels=labels_gen,
                                                                 temperature=args.temperature)
                
                # 3D VAE 解码: (B, C, D, H, W)
                sampled_images = vae.decode(sampled_tokens)
                # sampled_images = vae.decode(sampled_tokens / 0.2325) # 记得除回去！ 把 Latent 还原成 RGB。

        # measure speed after the first generation batch
        if i >= 1:
            torch.cuda.synchronize()
            used_time += time.time() - start_time
            gen_img_cnt += batch_size
            print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(gen_img_cnt, used_time, used_time / gen_img_cnt))

        if args.distributed:
            torch.distributed.barrier()
        sampled_images = sampled_images.detach().cpu()
        sampled_images = (sampled_images + 1) / 2

        # === 修改: 保存 3D 图片逻辑 ===
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            
            # sampled_images shape: (B, C, D, H, W)
            # 取出单张 3D 图: (C, D, H, W)
            vol = sampled_images[b_id].numpy()
            
            # --- 方案 A: 只保存中间切片为 PNG (为了快速查看) ---
            # 假设 Depth 是第 1 维 (Index 1)
            depth = vol.shape[1]
            mid_slice = vol[:, depth // 2, :, :] # (C, H, W)
            
            # 转换为 (H, W, C)
            gen_img = np.round(np.clip(mid_slice.transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)
            
            # BraTS 可能是 1 通道或 4 通道，cv2 需要 BGR
            if gen_img.shape[2] == 1: # 灰度图
                cv2.imwrite(os.path.join(save_folder, '{}_mid.png'.format(str(img_id).zfill(5))), gen_img)
            else: # 多通道，只存前3个通道或者转为RGB
                cv2.imwrite(os.path.join(save_folder, '{}_mid.png'.format(str(img_id).zfill(5))), gen_img[:, :, ::-1])

            # --- 方案 B: 保存完整 .nii.gz (如果安装了 nibabel) ---
            # if 'nib' in sys.modules:
            #     # (C, D, H, W) -> (D, H, W, C) for NIfTI standard
            #     nii_img = nib.Nifti1Image(vol.transpose(1, 2, 3, 0), np.eye(4))
            #     nib.save(nii_img, os.path.join(save_folder, '{}.nii.gz'.format(str(img_id).zfill(5))))

        # 2D 保存图片
        # distributed save
        # for b_id in range(sampled_images.size(0)):
        #     img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
        #     if img_id >= args.num_images:
        #         break
        #     gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
        #     gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
        #     cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    if args.distributed:
        torch.distributed.barrier()
    time.sleep(10)

    # back to no ema
    # 恢复现场
    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    # 3D 任务禁用 torch_fidelity 计算 (它只支持 2D ImageNet)
    # ToDo

    # 2D 版本
    # compute FID and IS
    # 计算指标 (FID & IS)
    # if log_writer is not None:
    #     if args.img_size == 256:
    #         input2 = None
    #         fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
    #     else:
    #         raise NotImplementedError
    #     metrics_dict = torch_fidelity.calculate_metrics(
    #         input1=save_folder,
    #         input2=input2,
    #         fid_statistics_file=fid_statistics_file,
    #         cuda=True,
    #         isc=True,
    #         fid=True,
    #         kid=False,
    #         prc=False,
    #         verbose=False,
    #     )
    #     fid = metrics_dict['frechet_inception_distance']
    #     inception_score = metrics_dict['inception_score_mean']
    #     postfix = ""
    #     if use_ema:
    #        postfix = postfix + "_ema"
    #     if not cfg == 1.0:
    #        postfix = postfix + "_cfg{}".format(cfg)
    #     log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
    #     log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
    #     print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
    #     # remove temporal saving folder
    #     shutil.rmtree(save_folder)

    if args.distributed:
        torch.distributed.barrier()
    time.sleep(10)


# 数据预处理逻辑：把数据集里的图片都跑一遍 VAE 编码器，得到 Latent 并保存到硬盘。
def cache_latents(vae,
                  data_loader: Iterable,
                  device: torch.device,
                  args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Caching: '
    
    print_freq = 1
    print("Starting cache_latents loop...")

    for data_iter_step, (samples, _, paths) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        t_start = time.time()

        samples = samples.to(device, non_blocking=True)
        t_data = time.time()

        # === 调试输出: 打印一下形状，确保是 3D 的 (B, C, D, H, W) ===
        # if data_iter_step == 0:
        #     print(f"[Debug] Input samples shape: {samples.shape}")

        with torch.no_grad():

            # 1. 正常编码
            posterior = vae.encode(samples)
            moments = posterior.parameters # 获取均值和方差

            t_encode1 = time.time() # 第一遍编码完成时间

            # 2. 翻转编码 (Flipping Trick)

            # 3D 版本
            # 2D: (B, C, H, W) -> flip dims=[3] (Width)
            # 3D: (B, C, D, H, W) -> flip dims=[4] (Width)
            posterior_flip = vae.encode(samples.flip(dims=[4]))
            moments_flip = posterior_flip.parameters

            # 2D 版本
            # 顺便把图片水平翻转后的 Latent 也算出来了。
            # posterior_flip = vae.encode(samples.flip(dims=[3]))
            # moments_flip = posterior_flip.parameters

            t_encode2 = time.time() # 第二遍编码完成时间

        # 3. 存盘
        for i, path in enumerate(paths):
            save_path = os.path.join(args.cached_path, path + '.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, moments=moments[i].cpu().numpy(), moments_flip=moments_flip[i].cpu().numpy())

        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()

        t_end = time.time()
        
        # === 调试输出: 打印耗时详情 ===
        print(f"[Step {data_iter_step}] "
              f"Data: {t_data - t_start:.2f}s | "
              f"Encode1: {t_encode1 - t_data:.2f}s | "
              f"Encode2: {t_encode2 - t_encode1:.2f}s | "
              f"Save: {t_end - t_encode2:.2f}s | "
              f"Total: {t_end - t_start:.2f}s")
    return
