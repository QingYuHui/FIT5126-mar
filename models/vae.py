# Adopted from LDM's KL-VAE: https://github.com/CompVis/latent-diffusion
import torch
import torch.nn as nn

import numpy as np


# Swish 激活函数, 非线性
def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

# 归一化层, 使用 Group Norm
#
# Batch Norm (BN, 批归一化), 沿着 Dimension (N, H, W) 计算均值和方差, 
# 对每个通道进行归一化, 如果此时 batch_size=1, 则无法计算均值和方差, 因为只有一个样本
# 但是在大 batch_size 下效果很好
#
# Group Norm (GN, 分组归一化), 把通道分成若干组, 它不依赖 Batch Size，N=1 和 N=100 算出来的结果是一样的
# 然后沿着 Dimension (C_group, H, W) 计算均值和方差, 对每个样本进行归一化
#
def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )

# 上采样, 放大图像
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        # 1. 插值放大：把图片长宽放大 2 倍，使用"最近邻"插值
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        # 2. 卷积（可选）：平滑放大后的像素
        if self.with_conv:
            x = self.conv(x)
        return x

# 下采样, 缩小图像
class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv3d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        # 1. 卷积模式： 使用步幅为 2 的卷积直接减半尺寸
        if self.with_conv:
            # pad = (0, 1, 0, 1) # 2D 版本

            # 3D 版本顺序是: (W_left, W_right, H_top, H_bottom, D_front, D_back)
            pad = (0, 1, 0, 1, 0, 1)

            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            # 2. 池化模式：直接用平均池化减半尺寸
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


# 残差块 (ResNet Block)
class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        # 包含两个卷积层 (conv1, conv2)，两个归一化层 (norm1, norm2)
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            # 把时间向量投影进来的线性层 (temb_proj)
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        # 如果输入通道和输出通道不一样（比如输入64变输出128），x 和 h 没法直接相加。
        # 这时就需要一个 1x1 或 3x3 的卷积把 x 的通道数也变一下，对齐尺寸
        # !!! 这里没有关闭 bias !!!, 不清楚为什么
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    # 假设:
    # 图像 x: [Batch=1, Channel=64, Height=32, Width=32]
    # temb: [Batch=1, temb_channels=512]
    def forward(self, x, temb):

        # 阶段 A：第一层卷积与“时间注入”
        h = x
        h = self.norm1(h)   # 1. 归一化 (GroupNorm)
        h = nonlinearity(h) # 2. 激活 (Swish)
        h = self.conv1(h)   # 3. 卷积 (Conv2D), 提取特征

        if temb is not None:
            # 4. 注入时间信息！
            # temb_proj 把时间向量从 512 变到 64 (跟 h 的通道数对齐)
            # [:, :, None, None] 是把 (B, C) 变成 (B, C, 1, 1)，以便广播加到整张图上
            # h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None] # 3D 版本

        # 阶段 B：第二层卷积, 在融合了时间信息后，再进行一次精细的特征处理。
        h = self.norm2(h)   # 5. 归一化 (GroupNorm)
        h = nonlinearity(h) # 6. 激活 (Swish)
        h = self.dropout(h) # 7. Dropout (防止过拟合，随机丢弃一点神经元)
        h = self.conv2(h)   # 8. 卷积 (Conv2D), 提取特征

        # 阶段 C：残差连接
        if self.in_channels != self.out_channels:
            # 如果通道数不匹配，调整 x 的通道数
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        # 最后把输入 x 和处理后的 h 相加，形成残差连接
        return x + h


# 全局自注意力层
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        # Query (查询)
        self.q = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        # Key (键)
        self.k = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        # Value (值)
        self.v = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):

        # 假设: x [Batch=1, Channel=64, Height=32, Width=32]。
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        # b, c, h, w = q.shape
        b, c, d, h, w = q.shape # 3D 版本

        # # 把 (32, 32) 的图拉成 1024 个点 (h*w)
        # q = q.reshape(b, c, h * w)
        q = q.reshape(b, c, d * h * w) # 3D 版本

        # 计算所有像素两两之间的关系
        q = q.permute(0, 2, 1)  # b,hw,c
        # k = k.reshape(b, c, h * w)  # b,c,hw
        k = k.reshape(b, c, d * h * w)  # b, c, dhw , 3D 版本

        # 矩阵乘法: [1, 1024, 64] * [1, 64, 1024] = [1, 1024, 1024]
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]

        # 缩放 (Scaled Dot-Product)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2) # 变成概率 (加起来为1)

        # attend to values
        # v: [1, 64, 1024]
        # w_: [1, 1024, 1024] (经过 permute)
        # v = v.reshape(b, c, h * w)
        v = v.reshape(b, c, d * h * w) # 3D 版本

        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        # h_ = h_.reshape(b, c, h, w) # 变回图片的形状 [1, 64, 32, 32]
        h_ = h_.reshape(b, c, d, h, w) # 恢复 3D 形状

        h_ = self.proj_out(h_) # 输出投影层

        return x + h_ # 残差连接 (原图 + 注意力提取的信息)

# 编码器, 把一张巨大的高清图片，压缩成一个尺寸很小、但信息高度浓缩的“潜在特征图” (Latent Feature Map)
# 输入：256 * 256 * 3 (一张高清 RGB 图片)
# 输出: 16 * 16 * 16 (一个高度浓缩的 Latent 特征图)
class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv3d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        # 设定每一层变几倍宽 (1, 1, 2, 2, 4)
        in_ch_mult = (1,) + tuple(ch_mult)

        self.down = nn.ModuleList()
        # 循环构建每一层
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                # 1. 堆叠 ResnetBlock (提取特征)
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                # 2. 如果当前分辨率在 attn_resolutions 名单里，就加 AttnBlock
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn

            # 3. 如果不是最后一层，就加 Downsample (把图片缩小一半)
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        # 再提取一次特征
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        # 全局注意力 (此时图片很小)
        # self.mid.attn_1 = AttnBlock(block_in)
        # 再提取一次特征
        # self.mid.block_2 = ResnetBlock(
        #     in_channels=block_in,
        #     out_channels=block_in,
        #     temb_channels=self.temb_ch,
        #     dropout=dropout,
        # )

        # end

        # 允许模型在训练时通过重参数化技巧（Reparameterization Trick）进行采样
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    # 例如:
    # 输入图片: 256 x 256 RGB (3通道), 基础通道ch=128
    # 通道倍率: (1, 2, 4), z_channels:4, double_z=True
    # conv_in: [1,3,256,256] -> [1,128,256,256]
    # downsample 1: [1,128,256,256] -> [1,128,128,128]
    # downsample 2: [1,256,128,128] -> [1,256,64,64]
    # downsample 3: [1,512,64,64] -> [1,512,32,32]
    # middle: [1,512,32,32] -> [1,512,32,32], 在 32 x 32 的图上做 Attention。因为图小，计算量可以接受。
    # End: [1,512,32,32] -> [1,8,32,32] (z_channels=4, double_z=True)
    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        # h = self.mid.attn_1(h)
        # h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


# 拿着 Encoder 给它的那个高度浓缩的“压缩包”（Latent Z），把它一层层放大，最后还原成一张 256x256 的高清 RGB 图像。
# 输入: 16 * 16 * 16 (一个高度浓缩的 Latent 特征图)
# 输出：256 * 256 * 3 (一张高清 RGB 图片)
class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        give_pre_end=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1] # 计算最底层的通道数，比如 128 * 4 = 512
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # self.z_shape = (1, z_channels, curr_res, curr_res)
        self.z_shape = (1, z_channels, curr_res, curr_res, curr_res) # 3D 版本
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        # z to block_in: 把 4 通道变回 512 通道
        self.conv_in = torch.nn.Conv3d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        # self.mid.attn_1 = AttnBlock(block_in)
        # self.mid.block_2 = ResnetBlock(
        #     in_channels=block_in,
        #     out_channels=block_in,
        #     temb_channels=self.temb_ch,
        #     dropout=dropout,
        # )

        # upsampling
        self.up = nn.ModuleList()
        # 上采样层 (self.up) —— 核心的放大过程
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1): # 更深
                # 1. 堆叠 ResnetBlock (细化特征)
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                # 2. 如果需要，加 Attention
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            # 3. 关键：Upsample (放大图片)
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        # 输出层 (end)
        # conv_out：把 128 个特征通道，压缩成我们最终想要的 out_ch=3 (R, G, B) 通道，形成最终的图像。
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        # h = self.mid.attn_1(h)
        # h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        # 这个开关允许 Decoder 不输出 RGB 图，而是直接把最后一层的特征图扔出来，供 Loss Function 计算使用。
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


# 对角高斯分布 (Diagonal Gaussian Distribution) 的构建、采样和损失计算。
# 简单来说，这是一个**“概率转换器”。它把神经网络输出的死板数值，转换成了一个概率分布**，并从中采样出 Latent Code (潜在编码)。
# 它是连接 Encoder（编码器）和 Decoder（解码器）的桥梁。
class DiagonalGaussianDistribution(object):

    # parameters 是 Encoder 最后一层的输出（例如 Conv_out 输出的 8 通道特征图）
    # 切分：它把这 8 个通道一刀两断，前 4 个通道作为 均值 (Mean)，后 4 个通道作为 对数方差 (Log-Variance)
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        # 1. 切分：把输入切成两半, 前 16 个通道作为 均值，后 16 个通道作为 对数方差 (Log-Variance)
        # 因为方差必须是正数，神经网络直接输出正数很难控制。输出对数（可以是负无穷到正无穷），然后取 exp，就天然保证了方差一定是正数
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        # 2. 钳位：防止数值爆炸, 数值稳定, 防止 exp 运算后数值变成 NaN 或无穷大
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        # ... 计算标准差和方差 ...
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    # 重参数化技巧 (The Magic Trick)
    # 如果在神经网络中间直接搞一个“随机抽签”的操作，反向传播时梯度是断掉的（你没法对“随机数”求导）
    # 重参数化技巧的核心思想是：把随机采样的过程，变成一个确定性的函数加上一个独立的随机噪声。
    # z = mean + std * ε，其中 ε ~ N(0, I) 是一个独立的标准正态分布噪声。
    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    # 计算 KL 散度 (Kullback-Leibler Divergence)
    # 作用：这是一个损失函数 (Loss Function) 的一部分。
    # 目的：它强制要求 Encoder 预测出来的分布，尽量接近 标准正态分布 N(0, I)。
    #
    # 如果没有这个限制，Encoder 会为了降低重构误差，把方差缩得非常小（接近 0），把均值散得非常开，
    # 这样 VAE 就退化成了普通的 Autoencoder，失去了生成新数据的能力。
    # KL Loss 就像一根绳子，把分布拉回原点，保持潜在空间的平滑性。
    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            # 2D 数据: [Batch, C, H, W] -> dims=[1, 2, 3]
            # 3D 数据: [Batch, C, D, H, W] -> dims=[1, 2, 3, 4]
            dims = [1, 2, 3, 4]
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=dims,
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=dims,
                )

    # nll: 计算负对数似然 (Negative Log-Likelihood)
    # 衡量“某一个样本”属于“当前这个高斯分布”的概率有多大。 
    # 或者反过来说：如果这个样本真的是从这个分布里出来的，我会觉得有多“惊讶”？惊讶度越高，NLL 值越大，Loss 越大。
    # def nll(self, sample, dims=[1, 2, 3]):
    def nll(self, sample, dims=[1, 2, 3, 4]): # 3D 版本
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean
    

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer


# “图像压缩机” 和 “解压机” 的角色。
# 它负责把庞大的 RGB 图像压缩成 16x16x16(举例) 的 Latent 特征（Token），以及把 Latent 还原回图像
class AutoencoderKL(nn.Module):
    # embed_dim: 潜变量维度 (Latent Dimension), 它决定了压缩后的特征图在这个维度上有多少层
    # ch_mult: 通道倍率, 这是一个列表（如 [1, 1, 2, 2, 4]），控制 Encoder/Decoder 中每一层卷积通道数的膨胀倍数。它决定了网络的深度和宽度
    # use_variational: 是否使用变分推理。默认为 True。这意味着它是一个 VAE，而不是普通的 Autoencoder。它学习的是一个概率分布（高斯分布），而不是固定的数值。
    # ckpt_path: 预训练权重路径。如果有路径，初始化时直接加载权重（因为 MAR 训练时这个 VAE 是冻结不动的）
    # def __init__(self, embed_dim, ch_mult, use_variational=True, ckpt_path=None, num_res_blocks=2):


    # ddconfig: 包含 encoder/decoder 配置的字典
    # n_embed: 词表大小 (8192)
    # embed_dim: 潜在维度 (4)
    def __init__(self, ddconfig, n_embed, embed_dim, ckpt_path=None):
        super().__init__()

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        # self.use_variational = use_variational
        self.embed_dim = embed_dim

        # 说明:
        # 
        # VAE (Variational Autoencoder) —— 变分模式
        # 原理：VAE 认为潜在空间是一个高斯分布（正态分布）。为了描述这个分布，它需要两个参数：均值 ($\mu$) 和 方差 ($\sigma$)。
        # 通道数：如果你设定的 Latent 维度是 4，Encoder 必须输出 4+4=8 个通道（前4个是均值，后4个是方差）。
        #
        # VQGAN (Vector Quantized GAN) —— 向量量化模式
        # 原理：VQGAN 的 Encoder 只是把图片压缩成一个固定的、确定性的特征向量。它不需要采样，也不需要预测分布。
        # 通道数：如果你设定的 Latent 维度是 4，Encoder 就直接输出 4 个通道。

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        # # 2. 定义 quant_conv (关键修改点)
        # if self.use_variational:
        #     # VAE 模式: Encoder 输出均值+方差 (2 * embed_dim) -> 映射到 (2 * embed_dim)
        #     self.quant_conv = torch.nn.Conv3d(2 * embed_dim, 2 * embed_dim, 1)
        # else:
        #     # VQGAN 模式: Encoder 输出 Latent (embed_dim) -> 映射到 (embed_dim)
        #     self.quant_conv = torch.nn.Conv3d(my_z_channels, my_z_channels, 1)
        # self.post_quant_conv = torch.nn.Conv3d(my_z_channels, my_z_channels, 1)

        # --- 替换为 (VQGAN 标准写法) ---
        # 确保 z_channels 和 embed_dim 对齐 (通常都是 4)
        self.quant_conv = torch.nn.Conv3d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        # 2D 版本
        # sd = torch.load(path, map_location="cpu")["model"]

        ckpt = torch.load(path, map_location="cpu")

        # 3D 版本, 兼容性处理
        if "model" in ckpt:
            sd = ckpt["model"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt

        msg = self.load_state_dict(sd, strict=False)
        print("Loading pre-trained KL-VAE")
        print("Missing keys:")
        print(msg.missing_keys)
        print("Unexpected keys:")
        print(msg.unexpected_keys)
        print(f"Restored from {path}")

    # 把 RGB 变成 Latent 的过程
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        # VQGAN 的 encode 不仅返回 latent，还返回 loss 和 info
        # 我们推理时只需要第一个返回值
        quant, emb_loss, info = self.quantize(h)
        return quant

    # 把 Latent 变成 RGB 的过程
    def decode(self, z):
        quant = self.post_quant_conv(z)
        dec = self.decoder(quant)
        return dec

    def forward(self, inputs, disable=True, train=True, optimizer_idx=0):
        quant = self.encode(inputs)
        dec = self.decode(quant)
        return dec

# -------------------------------------------------------------------
# 第一阶段（VAE 负责）：搞定“视觉”
# 这部分负责把图片看清楚，压缩好，再还原回去。
# 现状：这步通常是别人已经做好的。
# 你加载的那个 ckpt_path 就是别人在大规模数据上训练了好几周的成果。
# 它已经学会了怎么把猫毛、草地纹理压缩进 $z$ 里。
# 你的操作：加载它 $\to$ 冻结它 (Frozen) $\to$ 只当个 API 用。
#
# 第二阶段（你的模型负责）：搞定“逻辑”
# 这部分负责学习“画什么”。比如“一只猫坐在沙发上”，
# 在潜在空间 ($z$) 里长什么样。
# 你的操作：这就是你要写的核心代码（比如 Transformer 或 UNet）。
# 它只看得到 $z$，根本看不到原始像素。
#
# -------------------------------------------------------------------
# VAE 是“翻译官”：
# 它懂“像素语言”（繁琐、巨大）和“潜码语言 ($z$)”（精简、抽象）。
# 它的工作只是把巨大的像素书翻译成精简的 $z$ 笔记，或者反过来。
# 你不需要教翻译官怎么翻译，因为他是专业的（Pre-trained）。
#
# 你要训练的模型是“作家”：
# 作家不用懂像素语言，他只需要读懂 $z$ 笔记。
# 作家的任务是：根据上文，写出下一段精彩的 $z$ 笔记。
# 写完后，丢给翻译官（Decoder）去打印出来就行了。
#
# -------------------------------------------------------------------
# 需要关心的是 $z$ 的特性：
# $z$ 的形状：
# 要知道经过 VAE 压缩后，数据变成了什么样子。
# 例如：输入 $256 \times 256 \times 3 \to$ 输出 $32 \times 32 \times 4$。
# 你的模型 Input/Output 必须匹配这个 $32 \times 32 \times 4$。
#
# Scale Factor (缩放因子)：
# 这是一个巨坑。VAE 输出的 $z$ 通常数值很小（比如在 -1 到 1 之间，甚至更小）。
# 为了让你的 Diffusion/Transformer 更好训练，通常会把 $z$ 乘以一个常数（比如 0.18215 或 0.2325）把它放大到标准正态分布的范围。
# 你要关心的是：在送入你的模型前，有没有乘这个数？出来后有没有除回去？
#
# Tokenization (如果你做的是 VAR/GPT 类模型)：
# 如果你的模型是处理离散序列的（像 GPT），你需要关心怎么把这个连续的 $z$ 变成离散的整数索引 (Quantization / Codebook)。
#
