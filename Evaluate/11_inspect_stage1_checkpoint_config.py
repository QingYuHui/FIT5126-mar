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

# 检查 VQGAN Stage 1 Checkpoint 的结构
# 以推断模型配置参数 (n_embed, embed_dim, z_channels, ch, ch_mult 等等)

"""
Purpose:
    Inspect the VQGAN/VAE stage-1 checkpoint structure and infer key model
    settings such as codebook size, embedding dimension, latent channels,
    base channel count, and approximate downsampling depth.

Suggested filename:
    01_inspect_stage1_checkpoint_config.py
"""

import torch

CKPT_PATH = "vqgan/stage1.ckpt"

def analyze_vqgan_ckpt(path):
    print(f"正在分析: {path} ...")
    try:
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
            print("✅ 这是一个 PyTorch Lightning Checkpoint (包含 'state_dict')")
        else:
            print("✅ 这是一个原生 PyTorch State Dict")
            
        print("-" * 30)
        
        # --- 侦探工作开始 ---
        
        # 1. 寻找 Codebook (量化器)
        # 通常 key 叫做 'quantize.embedding.weight'
        if 'quantize.embedding.weight' in sd:
            shape = sd['quantize.embedding.weight'].shape
            n_embed = shape[0]
            embed_dim = shape[1]
            print(f"🕵️ [关键线索] Codebook:")
            print(f"   - n_embed (词表大小): {n_embed}")
            print(f"   - embed_dim (嵌入维度): {embed_dim}")
        else:
            print("⚠️ 未找到 quantize.embedding.weight，可能是旧版模型或键名不同。")

        # 2. 寻找 Z-Channels (潜空间通道数)
        # 通常在 'post_quant_conv' 或 'quant_conv'
        if 'post_quant_conv.weight' in sd:
            # shape: [embed_dim, z_channels, 1, 1]
            shape = sd['post_quant_conv.weight'].shape
            z_channels = shape[1]
            print(f"🕵️ [关键线索] Z-Channels (Latent Channels): {z_channels}")
            
            if shape[0] != embed_dim:
                 print(f"⚠️ 注意: post_quant_conv 输出 ({shape[0]}) 与 embed_dim ({embed_dim}) 不匹配，请检查配置。")

        # 3. 寻找 Base Channels (基础通道数)
        # 也就是 model 的 ch 参数
        if 'decoder.conv_in.weight' in sd:
            # shape: [ch, z_channels, 3, 3]
            shape = sd['decoder.conv_in.weight'].shape
            ch = shape[0]
            print(f"🕵️ [关键线索] Base Channels (ch): {ch}")

        # 4. 推测 Downsampling 层数 (ch_mult)
        # 我们通过计算 decoder 中 up 层的数量来推测
        up_blocks = [k for k in sd.keys() if 'decoder.up' in k and '.block' in k]
        # decoder.up.0, decoder.up.1 ... 最大的数字
        if up_blocks:
            max_idx = 0
            for k in up_blocks:
                parts = k.split('.')
                try:
                    idx = int(parts[2]) # decoder.up.X...
                    if idx > max_idx: max_idx = idx
                except: pass
            # 层数通常是 max_index + 1
            print(f"🕵️ [推测线索] Downsample Levels (可能对应 ch_mult 长度): {max_idx + 1}")

        # 5. 检查输入通道
        if 'encoder.conv_in.weight' in sd:
             in_ch = sd['encoder.conv_in.weight'].shape[1]
             print(f"🕵️ [关键线索] 输入图片通道数 (in_channels): {in_ch} (3代表RGB)")

    except Exception as e:
        print(f"❌ 读取失败: {e}")

analyze_vqgan_ckpt(CKPT_PATH)
