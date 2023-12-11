# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:57:57 2023

@author: User
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from time import time
from scipy.optimize import linear_sum_assignment as linear_assignment

# 定義計算SSE的函數
def calculate_sse(X, labels, centers):
    distances = np.linalg.norm(X - centers[labels], axis=1)
    return np.sum(distances**2)

# 定義計算層次聚類和DBSCAN SSE的函數
def calculate_non_centroid_sse(X, labels):
    unique_labels = set(labels)
    if -1 in unique_labels:  # 忽略DBSCAN的噪聲點
        unique_labels.remove(-1)
    sse = 0
    for k in unique_labels:
        cluster_k = X[labels == k]
        center_k = cluster_k.mean(axis=0)
        sse += np.sum((cluster_k - center_k) ** 2)
    return sse

# 定義計算熵的函數
def calculate_entropy(labels):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=np.e)

# 讀取數據
df = pd.read_csv('sizes3 (with class label).csv')

# 準備數據
X = df[['x', 'y']].values
y_true = df['class'].values

# 標準化特徵
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 定義分群算法
cluster_algorithms = {
    'KMeans': KMeans(n_clusters=4, random_state=42),
    'Hierarchical': AgglomerativeClustering(n_clusters=4),
    'DBSCAN': DBSCAN(eps=0.3, min_samples=10)  
}
# 'eps' 和 'min_samples' 的值 嘗試不同的參數設定，並且比較分群結果

# 定義函數來找到最佳的標籤映射
def best_label_mapping(true_labels, pred_labels):
    D = max(true_labels.max(), pred_labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(pred_labels.size):
        w[pred_labels[i], true_labels[i]] += 1
    ind = linear_assignment(-w)
    return ind

# 分群和性能評估
for name, algorithm in cluster_algorithms.items():
    start_time = time()
    
    # 分群
    cluster_labels = algorithm.fit_predict(X_scaled)
    end_time = time()
    
    # 計算SSE
    sse = None
    if hasattr(algorithm, 'inertia_'):
        sse = algorithm.inertia_
    elif name in ['Hierarchical', 'DBSCAN']:
        sse = calculate_non_centroid_sse(X_scaled, cluster_labels)

    # 計算Accuracy
    ind = best_label_mapping(y_true, cluster_labels)
    new_labels = np.zeros_like(cluster_labels)
    for i in range(len(ind[0])):
        new_labels[cluster_labels == ind[1][i]] = ind[0][i]
    accuracy = accuracy_score(y_true, new_labels)

    # 計算Entropy
    ent = calculate_entropy(cluster_labels)

    # 打印性能指標
    print(f'{name} Clustering')
    print(f'    Time taken: {end_time - start_time:.4f} seconds')
    if sse is not None:
        print(f'    SSE: {sse:.4f}')
    print(f'    Accuracy: {accuracy:.4f}')  # 输出准确性
    print(f'    Entropy: {ent:.4f}')
    print('--------------------------------------------------')
    
    # 繪製分群結果
    plt.figure(figsize=(8, 4))
    markers = ['1', '2', '3', '4']  #定義標籤
    for i, marker in zip(range(4), markers):
        plt.plot(X_scaled[cluster_labels == i, 0], X_scaled[cluster_labels == i, 1], 
                 marker=marker, linestyle='', label=f'Cluster {i+1}')
    plt.title(f'{name} Clustering Results')
    plt.xlabel('Feature 1 (standardized)')
    plt.ylabel('Feature 2 (standardized)')
    plt.legend()
    plt.show()