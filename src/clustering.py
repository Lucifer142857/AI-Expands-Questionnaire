import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def hierarchical_clustering(df_encoded):
    print("\n[2/7] 层次聚类分析 ...")
    X = df_encoded.values
    Z = linkage(X, method='ward')

    # 根据原始问卷数量计算最大聚类数，最少为2
    max_k = max(2, int(df_encoded.shape[0] / 20))
    for k in range(2, max_k + 1):
        labels = fcluster(Z, k, criterion='maxclust')
        counts = np.bincount(labels)[1:]
        print(f"  k={k} -> ", [f'簇{i+1}:{c}' for i, c in enumerate(counts)])

    plt.figure(figsize=(8,5))
    dendrogram(Z, truncate_mode='lastp', p=max_k, show_contracted=True)
    plt.title("层次聚类截断树状图")
    plt.xlabel("样本数")
    plt.ylabel("距离")
    plt.show()

    k = int(input(f"请输入最终聚类数 k (范围 2 到 {max_k}): "))
    return X, k, Z

def final_clustering(X, k):
    print("\n[3/7] 最终聚类 (KMeans & GMM) ...")
    # KMeans 保持 KMeans++ 初始化
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km_labels = km.fit_predict(X)
    sil_km = silhouette_score(X, km_labels) if k > 1 else 0

    # GMM 使用随机初始化和不同随机种子
    gmm = GaussianMixture(
        n_components=k,
        covariance_type='full',
        init_params='random',  # 使用随机初始化
        random_state=0         # 使用不同随机种子
    )
    gmm.fit(X)
    gmm_labels = gmm.predict(X)
    sil_gmm = silhouette_score(X, gmm_labels) if k > 1 else 0

    print(f"  KMeans 轮廓系数: {sil_km:.4f}")
    print(f"  GMM    轮廓系数: {sil_gmm:.4f}")

    if sil_gmm > sil_km:
        final_labels = gmm_labels
        centers = gmm.means_
        algo = "GMM"
        best_sil = sil_gmm
    else:
        final_labels = km_labels
        centers = km.cluster_centers_
        algo = "KMeans"
        best_sil = sil_km

    print(f"  采用算法: {algo}, 轮廓系数: {best_sil:.4f}")
    counts = np.bincount(final_labels)
    print("  各簇分布:", counts)

    return final_labels, centers
