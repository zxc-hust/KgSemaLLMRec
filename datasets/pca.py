import numpy as np
from sklearn.decomposition import PCA
import os

# 1. 加载原始嵌入矩阵
embeddings = np.load('datasets/amazon-book/llm_emb.npy')
print(f"原始嵌入矩阵形状: {embeddings.shape} (样本数×维度)")

# 2. 计算目标维度 (1536+64)/2 = 800
target_dim = 64
print(f"目标降维维度: {target_dim}")

# 3. 执行PCA降维
pca = PCA(n_components=target_dim, random_state=42)
reduced_embeddings = pca.fit_transform(embeddings)

# 4. 保存为NPY格式
output_dir = 'datasets/amazon-book'
os.makedirs(output_dir, exist_ok=True)

# 保存降维后的嵌入矩阵
np.save(os.path.join(output_dir, f'llm_emb_{target_dim}d.npy'), reduced_embeddings)
print(f"降维后矩阵形状: {reduced_embeddings.shape}")
print(f"保留方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")
print(f"结果已保存到 {output_dir}:")
# import matplotlib.pyplot as plt

# # 计算所有主成分
# full_pca = PCA().fit(embeddings)

# # 绘制方差累计曲线
# plt.figure(figsize=(10,6))
# plt.plot(np.cumsum(full_pca.explained_variance_ratio_), 'b-')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.axhline(y=0.95, color='r', linestyle='--')  # 标记95%阈值
# plt.grid()
# plt.savefig('pca_variance.png')  # 保存图像供分析

# # 自动选择保留95%方差的维度
# optimal_dim = np.argmax(np.cumsum(full_pca.explained_variance_ratio_) >= 0.95) + 1
# print(f"保留95%方差所需维度: {optimal_dim}")