import numpy as np

# 把路径换成你真实生成的一个 npz 文件路径
# file_path = "../cache/cache_128/BraTS2021_00000_t1.npz"
file_path = "../cache/cache_64/BraTS2021_00000_t1.npz"

# 加载缓存文件
data = np.load(file_path)

# 打印尺寸
print("原始 Latent 的尺寸:", data['moments'].shape)
print("翻转 Latent 的尺寸:", data['moments_flip'].shape)