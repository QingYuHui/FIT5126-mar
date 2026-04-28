from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Block

from models.diffloss import DiffLoss


# 根据 orders（每个样本的随机位置顺序）把前 mask_len 个位置标为 1（表示被 mask 的位置）
# 返回布尔型 mask (bsz, seq_len)
def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking

# 训练时把图像切成 token（patch），对一部分 token 做 mask，encoder 对已知 token 编码，
# decoder 将编码恢复到每个位置的条件向量 z，
# 然后用一个扩散式的 DiffLoss 来学习如何从这些 z 预测/重建被 mask 的原始连续 token。
# 推理时采用逐步去掩码（MaskGIT 风格的多轮策略）+ 扩散采样来生成完整图像 token，最后 unpatchify 得到图像。

# img_size: 输入图像的分辨率（默认 256）
# vae_stride：表示底层 VAE/encoder 的下采样 stride
# patch_size：表示每个 token 内部在 Latent Map 上聚合的大小
# seq_h = seq_w = img_size // vae_stride // patch_size：token 在高/宽方向上的数量
# seq_len = seq_h * seq_w：序列总长度（每张图的 token 数）
# vae_embed_dim：VAE 每个 token 的通道数（latent dim）, 默认 16
# token_embed_dim = vae_embed_dim * patch_size**2：每个 token 的总向量维度（patch 内元素展平后的维度）

# 尺寸假设:
# 输入图像：256 * 256 像素，RGB（3 通道）。
# VAE配置：vae_stride=16（下采样 16 倍），vae_embed_dim=16（16 个通道）。
# Patch配置：patch_size=1（即不进行额外的网格聚合，一个 latent 像素就是一个 token）。
# 
# 原始 RGB 图像的尺寸为 (3, 256, 256)。
# 经过 VAE 编码后，得到的 latent map 尺寸为 (16, 16, 16)。
# 空间尺寸缩小：256/16=16。通道数变为 Latent Dim：16。
#
# Patchify 后的 token 序列尺寸为 (L, D), (Sequence Length, Token Dimension)
# 此时 patch_size=1，所以 L=16*16=256，D=16*1*1=16。
# 
# 如果此时 patch_size=2，Patchify 会把每 2*2 个 Latent 像素聚合成一个 Token,
# 则 L=8*8=64，D=16*2*2=64。长度变短了，维度变高了。


# buffer_size 的作用
# Because the sampled sequence can be very short, 
# we always pad 64 [cls] tokens at the start of the encoder sequence, 
# which improves the stability and capacity of our encoding.
# 
class MAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 # class_num=1000,                # 数据集类别数
                 class_num=1,                     # BraTS 设为 1
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 depth_size=16
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size

        # self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        # self.seq_len = self.seq_h * self.seq_w

        # 修改：支持 3D 数据, 假设 img_size 是立方体 (D=H=W)
        self.seq_d = self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_d * self.seq_h * self.seq_w
        
        # self.token_embed_dim = vae_embed_dim * patch_size**2
        # 修改：支持 3D 数据
        self.token_embed_dim = vae_embed_dim * patch_size**3

        self.grad_checkpointing = grad_checkpointing

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):

        # 正态分布初始化
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    # (Patchify/Unpatchify) 把立体的 VAE Latent 特征图变成 Transformer 能吃的序列 (Token)

    # 3D 版本: (N, C, D, H, W) -> (N, L, Token_Dim)
    def patchify(self, x):
        # x: (B, C, D, H, W)
        bsz, c, d, h, w = x.shape
        p = self.patch_size
        
        # 确保尺寸能被 patch_size 整除
        assert d % p == 0 and h % p == 0 and w % p == 0
        
        d_, h_, w_ = d // p, h // p, w // p
        
        # 1. 维度切分 (关键步骤：切出 8 个维度)
        # 输入: (n, c, d, h, w)
        # 目标: (n, c, d_grid, p, h_grid, p, w_grid, p)
        x = x.reshape(bsz, c, d_, p, h_, p, w_, p)
        
        # 2. 维度重排 (把网格放一起，把Patch像素放一起)
        # n: batch
        # c: channel
        # u: d_grid (深度网格)
        # v: p (深度patch大小)
        # k: h_grid (高度网格)
        # x: p (高度patch大小)
        # y: w_grid (宽度网格)
        # z: p (宽度patch大小)
        # 目标: (n, d_grid, h_grid, w_grid, c, p, p, p)
        # 即: n u k y c v x z
        x = torch.einsum('ncuvkxyz->nukycvxz', x)
        
        # 3. 展平
        # L = d_ * h_ * w_
        # Dim = c * p * p * p
        x = x.reshape(bsz, d_ * h_ * w_, c * p ** 3)
        return x  # [N, L, Dim]

    # 3D 版本: (N, L, Token_Dim) -> (N, C, D, H, W)
    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        
        # 自动计算网格大小 (假设是立方体结构)
        # L = d_ * h_ * w_ -> 假设 d_=h_=w_ -> grid_size = L^(1/3)
        grid_size = int(round(x.shape[1] ** (1/3)))
        d_, h_, w_ = grid_size, grid_size, grid_size
        
        # 1. 恢复出 8 个维度
        # 输入: (N, L, Dim) -> (N, d_, h_, w_, c, p, p, p)
        x = x.reshape(bsz, d_, h_, w_, c, p, p, p)
        
        # 2. 维度重排 (修正了这里的 typo)
        # 输入: n u k y c v x z
        # 目标: n c u v k x y z  (即 Batch, Channel, d_grid, p, h_grid, p, w_grid, p)
        x = torch.einsum('nukycvxz->ncuvkxyz', x)
        
        # 3. 合并维度
        x = x.reshape(bsz, c, d_ * p, h_ * p, w_ * p)
        
        return x  # [N, C, D, H, W]

    # 为每个样本生成一个随机的排列顺序（Random Order）, 返回 (bsz, seq_len) 的随机置换序列
    # 假如序列有 256 个位置, 返回的每一行可能是 [34, 5, 200, 123, ..., 78] 这样的随机顺序
    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    # 根据随机顺序和采样到的掩码率，生成训练用的掩码
    def random_masking(self, x, orders):

        # 从分布中采样一个掩码率，假设 mask_ratio_generator 是 0.75
        # 意味着：256 个 token 中，我们要掩盖 192 个 token，只保留 64 个作为线索
        # 根据打乱的 orders，前 192 个位置标记为 1 (Masked)，后 64 个位置标记为 0 (Visible/Unmasked)
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding):

        # 如果是 5D 张量 [B, C, D, H, W]
        if x.dim() == 5:
            # 1. 把通道移到最后: [B, D, H, W, C]
            x = x.permute(0, 2, 3, 4, 1) 
            
            # 2. 【关键补丁】展平空间维度: [B, D*H*W, C]
            # 把中间的 D, H, W 三个维度拍扁成一个维度 (seq_len)
            x = x.flatten(1, 3)

        # 假如输入 Token 维度是 16。通过全连接层映射到 encoder_embed_dim (比如 1024)
        # 数据形状：[1, 256, 1024]
        # 投影 (z_proj)
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # 拼接 Buffer：在序列最前面加上 Class Embedding
        # 在序列最前面加一个代表“类别”的 token（例如“狗”）, 现在序列长度是 257
        #
        # 关于 buffer_size
        # mask 前面拼接 64 个 0，表示 buffer 部分永远是“可见”的
        # 无论图像被遮住多少，Encoder 至少 都能看到这 64 个 Class Tokens。这保证了 Attention 层总是有足够的数据进行计算
        
        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # 在训练时，以一定概率（label_drop_prob）将 Class Embedding 替换为可学习的 fake_latent
        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)

        # 加上位置嵌入
        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # 根据 mask，丢弃所有被掩盖的 token，只保留可见的 token 输入 Transformer
        # 假如原本有 257 个 token，根据之前的掩码，我们物理上删除了那 192 个被遮住的 token
        # dropping
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # 通过 encoder_blocks 提取特征
        # 输入 Encoder 的形状：[1, 65, 1024] (1 个 class + 64 个可见 token)
        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask):

        # 将编码器的输出（可见 token）和 mask_token（不可见位置）按原始位置拼接回去，恢复完整的序列长度
        # 编码器输出了 [1, 65, 1024] 的浓缩特征, 解码器需要还原回 256 个空间位置，才能进行图像生成
        # 代码首先创建了一个全 0 的序列（或者是可学习的 mask_token），长度为 256
        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # 将编码器输出的 64 个“真值”特征，填回到它们原始的索引位置（比如第 5、250、12 号位）
        # 剩下的 192 个空位，填入可学习的 mask_token（这就好比是占位符，代表“这里未知”）
        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # 加上解码器的位置嵌入
        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned

        # 现在数据形状变回了：[1, 257, 1024] (含 class token)

        # 通过 decoder_blocks 处理完整序列。
        # 解码器的任务是根据可见信息推断 mask 位置的信息
        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        # 解码器的输出并不是像素值，而是作为 Condition (条件)
        # 输出一个向量序列，这个序列不仅包含图像特征，还包含 class token

        # 最后加上 diffusion_pos_embed_learned，输出形状 [1, 256, 1024]（去掉了 class token）。这一步得到的变量在代码中叫 z
        
        # 切掉 buffer 部分（前 64 个 class token）
        # 切掉前 64 个，只保留图像部分
        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target, mask):

        # 传入 z (条件)：刚刚解码器猜出来的特征上下文 [1, 256, 1024]
        # target (真实 token): 最开始 VAE 编码得到的真实 Latent [1, 256, 16]

        # 它计算的是在条件 z 下，通过去噪过程预测 target 的误差。
        # 模型学习的是如何生成一个好的条件 z
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    # patchify → 随机 masking → encoder → decoder → diffloss loss
    def forward(self, imgs, labels):

        # class embed
        class_embedding = self.class_emb(labels)

        # print(f"[MAR Debug] Input images shape: {imgs.shape}")

        # 将图像 patchify 得到 gt_latents（Ground Truth）
        # patchify and mask (drop) tokens
        x = self.patchify(imgs)

        # 插入调试打印 🔥🔥🔥
        # print(f"[MAR Debug] After patchify shape: {x.shape}")

        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        # 跑 Encoder -> Decoder，得到预测的潜在向量 z
        # mae encoder
        x = self.forward_mae_encoder(x, mask, class_embedding)

        # mae decoder
        z = self.forward_mae_decoder(x, mask)

        # 调用 forward_loss 计算扩散损失
        # diffloss
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):

        # 一张全白的画布（全 mask），mask 全为 1
        # 例如设定总步数 num_iter = 100
        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            # class embedding and CFG
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)

            # 如果 cfg != 1.0，将输入复制一份，
            # 一份给有条件生成（带 Label），一份给无条件生成（带 Fake Latent），并在 Batch 维度拼接
            if not cfg == 1.0:
                # 1. 把 tokens 复制一份拼起来：[Token_Cond, Token_Uncond]
                tokens = torch.cat([tokens, tokens], dim=0)
                # 2. 把 class_embedding 拼起来：[真实标签, 假标签(fake_latent)]
                # 这就是核心！上半部分带着“狗”的标签，下半部分带着“空”的标签
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                # 3. Mask 也拼起来
                mask = torch.cat([mask, mask], dim=0)

            # Encoder-Decoder：运行一次模型，得到当前所有位置的条件向量 z
            # mae encoder
            x = self.forward_mae_encoder(tokens, mask, class_embedding)

            # mae decoder
            z = self.forward_mae_decoder(x, mask)

            # 使用余弦调度（Cosine Schedule）逐渐减少 mask 的比例。
            # 随着步数增加，mask 的区域越来越少，已知的区域越来越多
            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # 确定预测目标 (mask_to_pred)：
            # 计算当前步骤需要预测哪些 token（即上一轮是 mask，这一轮变成了非 mask 的那些位置）
            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # 采样 (self.diffloss.sample)
            # 提取需要预测位置的条件 z
            # CFG Scale：根据当前步数计算 CFG 的强度（支持 Linear 或 Constant 策略）
            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            
            # z 包含了 cond 和 uncond 的特征
            # cfg_iter 是当前这一步的 CFG 强度数值（比如 4.0）
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # 最后得到的 256*16 的序列，通过 unpatchify 变回 16*16*16 的 Latent 图
        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens


def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# -------------------------------------------------------------------
# 在潜在空间 (Latent Space) 上工作的图像生成模型。
# 它利用 MAE 的架构（Encoder 只看可见部分，Decoder 恢复全貌）来提取上下文特征，
# 然后用这些特征作为条件 (Condition)，去驱动一个扩散头 (Diffusion Head) 来生成被遮挡的图像部分。
#
# -------------------------------------------------------------------
# 训练流程 (forward 函数)
# 1. Masking (random_masking)：
# 随机生成一个掩码率（比如遮住 75%）。
# 随机打乱序列顺序 (sample_orders)。
# 把 75% 的 Token 挖掉，只留 25%。
# 
# 2. Encoder (forward_mae_encoder)：
# 只接收可见的 Token。
# 它加上了 class_embedding（类别标签，比如“生成一只狗”）和位置编码。
# 它的计算量很小，因为只处理 25% 的数据。
# 作用：理解现有的碎片是什么意思。
#
# 3. Decoder (forward_mae_decoder)：
# 把 Encoder 吐出来的特征放回原来的位置。
# 在空缺的位置填上可学习的 mask_token（占位符）。
# 处理完整的序列（256 个 Token）。
# 关键点：Decoder 输出的不是最终的像素/Latent值，而是一个Condition (条件向量 z)。
# 这个 z 包含了对缺失位置的“语义预测”（比如“这里应该是个眼睛”）。
#
# 4. DiffLoss (forward_loss)：
# MAR 把 Decoder 输出的 z 作为条件，输入到 self.diffloss。
# 本质：它在训练一个扩散模型，学习如何从 z 这个语义条件中，去噪得到真实的 Latent Token。
# 换句话说，DiffLoss 学习的是“如何根据上下文条件 z，生成被遮挡的图像部分”。
#
# -------------------------------------------------------------------
# 推理/生成流程 (sample_tokens 函数)
# 过程比喻： 想象你在画画，画布一开始是全白的。
# Step 0：全是空白 (Mask)。
# Step 1：模型看一眼空白，觉得大概要画个轮廓，于是用扩散模型生成了一些模糊的碎片（填补一部分 Mask）。
# Step 2：模型看着 Step 1 画出的碎片，觉得“哦，这好像是个猫头”，于是更有信心地画出了耳朵和胡须（再填补一部分 Mask）。
# Step N：不断重复，直到整张图画满。
#