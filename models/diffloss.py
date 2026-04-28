import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

from diffusion import create_diffusion


# DiffLoss: 扩散损失管理类
class DiffLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, grad_checkpointing=False):
        super(DiffLoss, self).__init__()
        self.in_channels = target_channels

        # 核心网络：SimpleMLPAdaLN, 干活的类
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,  # for vlb loss  # 因为模型通常需要同时预测噪声和方差（用于 VLB Loss），所以输出通道翻倍
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )

        # train_diffusion & gen_diffusion: 借用 OpenAI 的 diffusion 库来管理加噪和去噪的数学公式（比如什么时候加多少噪）。
        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")

    # 训练模式
    def forward(self, target, z, mask=None):

        # === 3D 数据维度说明 ===
        # target: [Total_Tokens, Token_Dim]
        #   Total_Tokens = Batch * (D*H*W) * Batch_Mul
        #   Token_Dim = vae_embed_dim * patch_size^3 (例如 16 * 1*1*1 = 16)

        # 2D 数据维度说明
        # target: 真实的 token 值（即 VAE 编码后的真值），Shape [1, 16]
        # z: AR 模型预测出的条件向量，Shape [1, 1024]
        # mask: 告诉 Loss 函数哪些 token 是这一轮需要预测的

        # 随机选一个时间步 t
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        
        # 它只计算 mask 部分的 Loss。也就是说，模型只学习如何画那些被遮住的部分，没被遮住的部分不用管。

        # （预测噪声 vs 真实加入的噪声）
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    # 推理模式
    def sample(self, z, temperature=1.0, cfg=1.0):

        # 输入：z (条件 [1, 1024])，cfg (Guidance Scale)
        # 初始噪声：生成纯高斯噪声 noise，Shape [1, 16]

        # diffusion loss sampling

        # 如果 cfg != 1.0（使用 Classifier-Free Guidance），
        # 它会把输入 noise 复制两份拼起来（一份给有条件生成，一份给无条件生成）
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            # 这里把 cfg 数值打了个包，准备传给下面
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        # 去噪循环 (p_sample_loop)
        # 每一步都调用 self.net，根据条件 z 把噪声一点点减掉
        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )

        return sampled_token_latent


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


# 时间步嵌入器
# 作用：把一个数字 t (比如 500) 变成一个 1024 维的向量。
# 原理：先用正弦/余弦函数（Sinusoidal）把标量扩展成高频/低频信号，再用 MLP 提取特征。
# 这让神经网络能敏锐地感知到“现在是去噪初期还是末期”。
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()

        # 特征提取 MLP
        # 虽然正弦波包含了丰富的信息，但它只是通用的数学特征，不一定适合当前这个特定的扩散模型。
        # 它把第一步生成的数学特征（frequency_embedding_size，比如 256维）投影到模型的主维度（hidden_size，比如 1024维）
        # 通过 Linear -> SiLU -> Linear 的非线性变换，让模型学会如何**“解释”**这些正弦波特征，提取出对去噪最有用的时间信息。
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    # 纯数学计算
    # 利用正弦和余弦函数（Sinusoidal）将标量时间 t 映射到一个向量空间
    # 如果直接输入数字 50 或 0.5，神经网络很难感知到“时间”的周期性和细微变化
    # 通过正弦/余弦变换，不同的 t 会被映射成频率不同的波形组合。
    # 这样，即使 t 发生微小变化，向量中某些高频分量也会发生显著变化，让网络对时间非常敏感
    # 输入：t (例如 [50], 标量)
    # 输出：t_freq (例如 [256], 向量)，包含了各种频率的正弦波特征。
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        # 1. 调制 (Modulation)
        # 平移, 缩放, 门控
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        # 2. 应用调制
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        # 3. MLP 计算
        h = self.mlp(h)
        # 4. 门控残差连接
        return x + gate_mlp * h


# 作用：把内部的隐层特征 (1024维) 变回输出维度 (32维)。
class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()

        # 通常 LayerNorm 自带可学习的 gamma (缩放) 和 beta (平移) 参数
        # 用 AdaLN 手动注入由条件 c 生成的 shift 和 scale, 所以这里禁用掉 elementwise_affine=False
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)

        # 小型的控制塔（SiLU + Linear）
        # 输入：条件 c (1024维)
        # 输出：2 * model_channels (shift 和 scale)(2048维)

        # linear
        # 最终的线性投影层
        # 特征维度 (1024) 压缩回输出维度 (32)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        # x (输入特征)：来自最后一个 ResBlock 的输出，形状 [1, 1024]。
        # 此时它已经包含了很多去噪信息，但数值分布可能还不是最终想要的
        # c (条件)：实际上是 y = t_emb + z_emb，即时间和AR上下文的混合体，形状 [1, 1024]

        # 条件 c 通过全连接层，输出 2048 个数值
        # 这些数值被拆分成两半，前 1024 个是 shift（平移量），后 1024 个是 scale（缩放量）
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)

        # norm_final 先把 x 的数值拉回标准正态分布（均值0，方差1），消除之前层的数值漂移
        x = modulate(self.norm_final(x), shift, scale)

        # 操作：[1, 1024] -> [1, 32]
        # 结果：输出了最终的张量。前 16 个数是预测的噪声 epsilon，后 16 个数是用于计算 VLB Loss 的方差插值系数
        x = self.linear(x)
        return x


# AdaLN (Adaptive Layer Normalization)：这是核心机制。
# 它不使用固定的 Norm 参数，而是根据条件 z 和时间 t 动态生成 Norm 的参数（shift 和 scale）。
# 这让条件 z 能直接控制每一层网络的统计分布。
class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # x: 当前的噪声 Token (比如 [1, 16])。
    # t: 当前的时间步 (比如 500)。(告诉模型现在噪点多不多)。
    # c: 条件向量 z (来自 MAR Transformer 的 [1, 1024] 的线索)。
    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        # (噪声 Token): [1, 16] -> 投影到 model_channels (比如 1024) -> [1, 1024]
        # t (时间 500): 变成向量 -> time_embed -> [1, 1024]
        # c (条件向量 z): [1, 1024] -> 线性变换 -> [1, 1024]

        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        # 融合条件, 将时间信息和上下文信息相加，得到总控制信号 y ([1, 1024])
        y = t + c

        # 经过多个 ResBlock，每个都使用 AdaLN 来调节
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        # 最后输出 [1, 32] (16 噪声 + 16 方差)
        return self.final_layer(x, y)

    # Classifier-Free Guidance
    def forward_with_cfg(self, x, t, c, cfg_scale):
        # x 包含了 [有条件样本, 无条件样本]
        # ... 跑一次网络 ...
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)

        # 拆分结果
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        # CFG 公式: Uncond + scale * (Cond - Uncond)
        # 意思是：在这个方向上，“有条件”比“无条件”多出来的特征，我要放大 scale 倍。
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

# -------------------------------------------------------------------
# 它的核心任务是：接收 Transformer 给出的“语义线索” (z)，通过去噪过程，
# 把随机噪声变成具体的 VAE Latent Token (target)。
# 和常见的 Stable Diffusion 不同，这里用的不是 UNet（处理 2D 图像），而是一个 MLP（多层感知机），
# 因为它处理的是单个 Token 的向量（1D 数据）。
#
#
#
#