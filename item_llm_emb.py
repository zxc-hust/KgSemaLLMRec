import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch

# 加载模型
model = SentenceTransformer("/data/LLM/bge-large-en-v1.5")

# 读取item_dict.json文件
with open("/data/zou/KGAT-pytorch-master/datasets/amazon-book/item_dict.json", "r") as f:
    item_dict = json.load(f)

# 按remap_id排序项目
sorted_items = sorted(item_dict.values(), key=lambda x: int(x["remap_id"]))

# 提取所有template
templates = [item["template"] for item in sorted_items]

# 批量生成嵌入向量
embeddings = model.encode(templates, normalize_embeddings=True, batch_size=32)

# PCA降维到256维
pca = PCA(n_components=256)
embeddings_pca = pca.fit_transform(embeddings)  # 输出为NumPy数组

# 保存为npz文件
np.savez("/data/zou/KGAT-pytorch-master/datasets/pretrain/amazon-book/item_llm_emb.npz", item_embed=embeddings_pca)

print(f"嵌入矩阵形状：{embeddings_pca.shape}")
print("嵌入已保存至 item_llm_emb.npz")