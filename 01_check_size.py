"""
Purpose:
    Inspect one cached latent .npz file and print the stored latent tensor
    shapes for the original and flipped volumes. Use this after running the
    latent caching script to confirm that cached latents have the expected
    shape, e.g. [4, 16, 16, 16] for 64^3 inputs with VAE stride 4.

Suggested filename:
    01_inspect_cached_latent_shape.py

Notes:
    Update file_path before running if your cache directory differs.
"""

import numpy as np

# 把路径换成你真实生成的一个 npz 文件路径
# file_path = "../cache/cache_128/BraTS2021_00000_t1.npz"
file_path = "output_cache/class0/BraTS2021_00000_t1.npz"

# 加载缓存文件
data = np.load(file_path)

# 打印尺寸
print("原始 Latent 的尺寸:", data['moments'].shape)
print("翻转 Latent 的尺寸:", data['moments_flip'].shape)
