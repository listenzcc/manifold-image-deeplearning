# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %%
class MatrixVisualizer:
    def __init__(self, n_clusters=3, random_state=None):
        """
        初始化可视化器
        
        参数:
            n_clusters: 聚类数量
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.random_state = np.random.randint(10, 500) if random_state is None else random_state
        self.scaler = StandardScaler()
        
    def reduce_dimension(self, X, method='tsne', n_components=2):
        """
        降维到2维或3维
        
        参数:
            X: 输入矩阵 (n_samples, 512)
            method: 降维方法 ['tsne', 'pca', 'mds', 'isomap']
            n_components: 降维后的维度 (2或3)
        
        返回:
            降维后的坐标
        """
        # 标准化数据
        X_scaled = self.scaler.fit_transform(X)
        
        if method == 'tsne':
            # t-SNE降维
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(30, X.shape[0] - 1),
                random_state=self.random_state,
                n_iter=1000,
                learning_rate='auto'
            )
            reduced = reducer.fit_transform(X_scaled)
            
        elif method == 'pca':
            # PCA降维
            reducer = PCA(n_components=n_components, random_state=self.random_state)
            reduced = reducer.fit_transform(X_scaled)
            
        elif method == 'mds':
            # MDS降维
            reducer = MDS(
                n_components=n_components,
                random_state=self.random_state,
                normalized_stress='auto'
            )
            reduced = reducer.fit_transform(X_scaled)
            
        elif method == 'isomap':
            # Isomap降维
            reducer = Isomap(
                n_components=n_components,
                n_neighbors=min(10, X.shape[0] - 1)
            )
            reduced = reducer.fit_transform(X_scaled)
            
        else:
            raise ValueError(f"未知的降维方法: {method}")
        
        return reduced
    
    def spectral_clustering(self, X, affinity='nearest_neighbors'):
        """
        谱聚类
        
        参数:
            X: 输入矩阵 (n_samples, 512)
            affinity: 相似度矩阵计算方法 ['nearest_neighbors', 'rbf']
        
        返回:
            聚类标签
        """
        # 标准化数据
        X_scaled = self.scaler.fit_transform(X)
        
        # 谱聚类
        spectral = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity=affinity,
            random_state=self.random_state,
            n_neighbors=min(10, X.shape[0] - 1) if affinity == 'nearest_neighbors' else None,
            gamma=1.0 # if affinity == 'rbf' else None
        )
        
        labels = spectral.fit_predict(X_scaled)
        return labels
    
    def visualize_2d(self, X, labels=None, method='tsne', title=None, figsize=(15, 12)):
        """
        2D可视化
        
        参数:
            X: 输入矩阵 (n_samples, 512)
            labels: 聚类标签 (如果为None则自动聚类)
            method: 降维方法
            title: 图像标题
            figsize: 图像大小
        """
        if labels is None:
            labels = self.spectral_clustering(X)
        
        # 降维到2D
        reduced_2d = self.reduce_dimension(X, method=method, n_components=2)
        
        # 创建图形
        fig = plt.figure(figsize=figsize)
        
        # 绘制散点图
        ax1 = plt.subplot(2, 2, 1)
        scatter = ax1.scatter(reduced_2d[:, 0], reduced_2d[:, 1], 
                             c=labels, cmap='tab10', s=50, alpha=0.8, 
                             edgecolors='w', linewidth=0.5)
        
        ax1.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax1.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        
        if title:
            ax1.set_title(f'{title} - 2D {method.upper()} Visualization', fontsize=14, fontweight='bold')
        else:
            ax1.set_title(f'2D {method.upper()} Visualization', fontsize=14, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Cluster', fontsize=12)
        
        # 绘制3D视图（可选）
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # 降维到3D
            reduced_3d = self.reduce_dimension(X, method=method, n_components=3)
            
            ax2 = fig.add_subplot(2, 2, 2, projection='3d')
            scatter_3d = ax2.scatter(reduced_3d[:, 0], reduced_3d[:, 1], reduced_3d[:, 2],
                                     c=labels, cmap='tab20', s=50, alpha=0.8,
                                     edgecolors='w', linewidth=0.5)
            
            ax2.set_xlabel(f'{method.upper()} 1', fontsize=10)
            ax2.set_ylabel(f'{method.upper()} 2', fontsize=10)
            ax2.set_zlabel(f'{method.upper()} 3', fontsize=10)
            ax2.set_title('3D View', fontsize=12, fontweight='bold')
            
        except ImportError:
            ax2 = plt.subplot(2, 2, 2)
            ax2.text(0.5, 0.5, '3D visualization requires matplotlib 3D toolkit',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('3D View (Not Available)', fontsize=12)
            ax2.axis('off')
        
        # 绘制不同降维方法的比较
        methods = ['tsne', 'pca', 'mds']
        colors = ['tab10', 'tab20', 'Set3']
        
        for idx, (viz_method, cmap) in enumerate(zip(methods[:2], colors[:2]), 3):
            try:
                ax = plt.subplot(2, 2, idx)
                reduced = self.reduce_dimension(X, method=viz_method, n_components=2)
                
                ax.scatter(reduced[:, 0], reduced[:, 1], 
                          c=labels, cmap=cmap, s=40, alpha=0.7,
                          edgecolors='w', linewidth=0.3)
                
                ax.set_xlabel(f'{viz_method.upper()} 1', fontsize=10)
                ax.set_ylabel(f'{viz_method.upper()} 2', fontsize=10)
                ax.set_title(f'{viz_method.upper()} Visualization', fontsize=11)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error with {viz_method}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return reduced_2d, labels
    
    def analyze_clusters(self, X, labels):
        """
        分析聚类结果
        
        参数:
            X: 输入矩阵
            labels: 聚类标签
        """
        print("=" * 50)
        print("聚类分析结果:")
        print("=" * 50)
        
        # 统计每个聚类的样本数
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        print(f"\n1. 聚类分布:")
        for label, count in zip(unique_labels, counts):
            percentage = count / len(labels) * 100
            print(f"   聚类 {label}: {count} 个样本 ({percentage:.1f}%)")
        
        # 计算每个聚类的中心
        print(f"\n2. 聚类中心特征:")
        for label in unique_labels:
            cluster_samples = X[labels == label]
            center = np.mean(cluster_samples, axis=0)
            std = np.std(cluster_samples, axis=0)
            print(f"   聚类 {label}: 均值范围 [{center.min():.3f}, {center.max():.3f}], "
                  f"标准差平均 {std.mean():.3f}")
        
        # 计算聚类间的距离
        print(f"\n3. 聚类间距离:")
        centers = []
        for label in unique_labels:
            centers.append(np.mean(X[labels == label], axis=0))
        
        centers = np.array(centers)
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                distance = np.linalg.norm(centers[i] - centers[j])
                print(f"   聚类 {i} 和 聚类 {j}: {distance:.3f}")
        
        print("=" * 50)


# 使用示例函数
def visualize_matrix(X, n_clusters=3, method='tsne', title=None):
    """
    快速可视化函数
    
    参数:
        X: n×512的矩阵
        n_clusters: 聚类数量
        method: 降维方法
        title: 图像标题
    """
    # 检查输入矩阵
    n_samples, n_features = X.shape
    print(f"输入矩阵形状: ({n_samples}, {n_features})")
    
    if n_features != 512:
        print(f"警告: 输入矩阵特征数为{n_features}，期望512")
    
    # 创建可视化器
    visualizer = MatrixVisualizer(n_clusters=n_clusters)
    
    # 执行谱聚类
    labels = visualizer.spectral_clustering(X)
    
    # 可视化
    reduced, labels = visualizer.visualize_2d(X, labels, method=method, title=title)
    
    # 分析聚类结果
    visualizer.analyze_clusters(X, labels)
    
    return reduced, labels


# 生成示例数据的函数（用于测试）
def generate_sample_data(n_samples=200, n_features=512, n_clusters=3):
    """
    生成示例数据
    
    参数:
        n_samples: 样本数量
        n_features: 特征数量（应为512）
        n_clusters: 真实聚类数量
    """
    np.random.seed(42)
    
    # 生成聚类中心
    centers = np.random.randn(n_clusters, n_features) * 2
    
    # 为每个中心生成样本
    X = []
    labels_true = []
    samples_per_cluster = n_samples // n_clusters
    
    for i in range(n_clusters):
        # 生成该聚类的样本
        cluster_samples = centers[i] + np.random.randn(samples_per_cluster, n_features) * 0.5
        X.append(cluster_samples)
        labels_true.extend([i] * samples_per_cluster)
    
    # 添加余数样本
    remainder = n_samples % n_clusters
    if remainder > 0:
        extra_samples = centers[0] + np.random.randn(remainder, n_features) * 0.5
        X.append(extra_samples)
        labels_true.extend([0] * remainder)
    
    X = np.vstack(X)
    labels_true = np.array(labels_true)
    
    # 打乱顺序
    indices = np.random.permutation(n_samples)
    X = X[indices]
    labels_true = labels_true[indices]
    
    return X, labels_true


# %%
# 主程序示例
# 生成示例数据
print("生成示例数据...")
# X, true_labels = generate_sample_data(n_samples=300, n_features=512, n_clusters=4)

files = open('./encoded.info').readlines()

times = [float(e.split('_')[-1].split('s')[0]) for e in files]

X = np.load('./encoded.npy')
plt.imshow(X)
plt.colorbar()
plt.show()

# %%

# 可视化数据
print("\n开始可视化分析...")
reduced_coords, cluster_labels = visualize_matrix(
    X, 
    n_clusters=10, 
    method='tsne',
    title="Spectural Clustering Visulization"
)

# %%
print(np.diff(cluster_labels))
for i in np.where(np.diff(cluster_labels))[0]:
    print(i, times[i])

# %%
plt.plot(times, cluster_labels)
plt.show()

# %%

# %%

# 可选：比较不同降维方法
print("\n\n比较不同降维方法:")
print("-" * 40)

visualizer = MatrixVisualizer(n_clusters=10)

methods = ['tsne', 'pca', 'mds']
for method in methods:
    print(f"\n使用 {method.upper()} 降维:")
    labels = visualizer.spectral_clustering(X)
    reduced = visualizer.reduce_dimension(X, method=method, n_components=2)
    
    # 简单绘制
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes[1]
    im = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=50, alpha=0.8)
    # for i, xy in enumerate(reduced):
    #     x, y = xy
    #     t = times[i]
    #     plt.text(x, y, f'{t}')
    ax.set_title(f'{method.upper()} + Clustering', fontsize=14)
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    plt.colorbar(im, ax=ax)

    ax = axes[0]
    im = ax.scatter(reduced[:, 0], reduced[:, 1], c=times, cmap='RdBu', s=50, alpha=0.8)
    ax.plot(reduced[:, 0], reduced[:, 1], alpha=0.1)
    ax.set_title(f'{method.upper()} + Clustering', fontsize=14)
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    plt.colorbar(im, ax=ax)

    fig.tight_layout()

    plt.show()

# %%