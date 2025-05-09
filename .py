import numpy as np

# 读取 .npz 文件
data = np.load("/data/zou/KGAT-pytorch-master/datasets/pretrain/amazon-book/llm_emb.npz")

# 获取嵌入矩阵（假设键名为 'item_embed'，根据实际保存的键名调整）
item_embeddings = data['item_embed']

# 输出形状
print("嵌入矩阵形状:", item_embeddings.shape)

