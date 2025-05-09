import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch

# 加载模型
model = SentenceTransformer("/data/LLM/bge-large-en-v1.5")

# 读取用户模板文件
with open("/data/zou/KGAT-pytorch-master/datasets/amazon-book/user_templates.json", "r") as f:
    user_templates = json.load(f)

# 按用户ID排序（确保与remap_id顺序一致）
sorted_users = sorted(user_templates.items(), key=lambda x: int(x[0]))
user_texts = [text for _, text in sorted_users]  # 提取模板文本

# 批量生成嵌入向量（与商品嵌入相同参数）
user_embeddings = model.encode(user_texts, normalize_embeddings=True, batch_size=32)

# PCA降维到256维
pca = PCA(n_components=256)
user_embeddings_pca = pca.fit_transform(user_embeddings)

# 保存为npz文件
np.savez("/data/zou/KGAT-pytorch-master/datasets/pretrain/amazon-book/user_llm_emb.npz", 
         user_embed=user_embeddings_pca)

print(f"用户嵌入矩阵形状：{user_embeddings_pca.shape}")
print("用户嵌入已保存至 user_llm_emb.npz")