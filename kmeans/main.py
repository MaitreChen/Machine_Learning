"""
Author: bin
Data:2022.10.23
"""
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_rand_score
from sklearn.metrics import mutual_info_score

from sklearn.metrics import silhouette_score

from plot_setting import *
from utils import *

from time import time


def data_loader(data_path, label_path, ratio):
    """
    随机取样时注意避免重复，另外，为了重现结果，这里采用了随机种子
    :param data_path: 样本数据
    :param label_path: 标签
    :param ratio: 从样本中随机选取的比例
    :return: 一定比例的数据及其标签
    """
    data_set, labels = load_mnist(label_path, data_path)

    np.random.seed(1)
    nums_of_data = data_set.shape[0]
    idx = np.random.choice(nums_of_data, int(nums_of_data * ratio), replace=False)

    return data_set[idx], labels[idx]


def dimension_reduction(data_set, mode='PCA', visualize=True):
    """
    数据降维，为了方便可视化与聚类的便捷性，将数据降到2维。之后可以考虑3维可视化
    :param data_set: 待降维的数据集
    :param mode: 降维方法选择，包括PCA、t-SNE两种方法，之后加入add LDA
    :param visualize: 是否启用可视化
    :return: 降维后的数据
    """
    data_reduced = None
    if mode == 'PCA':
        pca = PCA(n_components=2)
        data_reduced = pca.fit_transform(data_set)
        scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
        data_reduced = scale.fit_transform(data_reduced)
    elif mode == 't-SNE':
        ts = TSNE(n_components=2, random_state=0, verbose=1)
        data_reduced = ts.fit_transform(data_set)
        scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
        data_reduced = scale.fit_transform(data_reduced)
        data_reduced = np.array(data_reduced, 'double')
    elif mode == 'LDA':
        pass

    # 可视化
    if visualize:
        plt.figure(figsize=(14, 10))
        plt.title(f"Description and Visualization with {mode}", fontsize=25)
        for i in range(10):
            plt.scatter(data_reduced[label == i, 0], data_reduced[label == i, 1], c=color_set[i], marker=mark[i],
                        label=i)

        plt.xlabel('Feature 1', fontsize=18)
        plt.ylabel('Feature 2', fontsize=18)
        plt.legend()
        plt.show()

    return data_reduced


# 初始化簇中心
def init_cluster_center(data_set, k):
    init_center = np.random.choice(len(data_set), k, replace=False)
    centers = data_set[init_center]
    return centers


# 计算每一个点与簇中心(center)的距离，找到最近的center，存储到assign_set中
def assign_points(data_set, centers, k):
    assign_set = [[] for _ in range(k)]
    for point in data_set:
        dist_set = [euclidean_distance(point, center) for center in centers]
        min_dist = np.min(dist_set)  # 找到距离最小值
        index = np.where(dist_set == min_dist)[0][0]  # 找到距离最小值对应的索引
        assign_set[index].append(point)
    assign_set = np.array(assign_set)
    return assign_set


# 更新簇中心,将每一个簇中（assign_set）所有点的横、纵坐标的均值作为均值
def update_cluster_center(center, assign_set, k):
    delta = 0  # 计算变化量，迭代终止条件
    for i in range(k):
        sum_X = 0
        sum_Y = 0
        length = len(assign_set[i])
        if length == 0:
            continue
        for j in range(length):
            sum_X += assign_set[i][j][0]
            sum_Y += assign_set[i][j][1]

        new_center = (sum_X / length, sum_Y / length)
        delta += euclidean_distance(center[i], new_center)
        center[i] = new_center

    return center, delta


def compute_sse(assign_set, center):
    """
    作为聚类的评价指标，同时也是为了得到最佳的k值
    当然，实际生活中应当结合需求选择k值，而不单单是通过簇内误方差
    :param assign_set:每一个簇集合
    :param center:簇心
    :return:簇内误方差
    """
    error = 0
    for i in range(len(assign_set)):
        for j in range(len(assign_set[i])):
            error += euclidean_distance(assign_set[i][j], center[i])
    return error


def k_means(data_set, k):
    init_center = init_cluster_center(data_set, k)
    assign_set = assign_points(data_set, init_center, k)
    new_center, delta = update_cluster_center(init_center, assign_set, k)

    while delta != 0:  # 当所有簇心不再变化，迭代结束
        new_center, delta = update_cluster_center(init_center, assign_set, k)
        assign_set = assign_points(data_set, new_center, k)
    return assign_set, new_center


def plot_sse(data_set, lower, higher):
    sse_set = []
    for i in range(lower, higher + 1):
        t_cluster, t_cluster_center = k_means(data_set, i)
        sse = compute_sse(t_cluster, t_cluster_center)
        sse_set.append(sse)
        print(f"n_clusters{i}   sse:{sse}")
    plt.title('The relationship between n_clusters and SSE', fontsize=18)
    plt.plot(range(lower, higher + 1), sse_set, color='b', marker='o')
    plt.xlabel('n_clusters', fontsize=14)
    plt.ylabel('SSE', fontsize=14)
    plt.xticks(range(1, 11))
    plt.show()


# Visualization Result of cluster
def plot_cluster(assign_set, center, k):
    plt.figure(figsize=(14, 10))
    plt.title(f"Cluster of K-Means(n_cluster={k})", fontsize=25,
              bbox=dict(facecolor='y', edgecolor='blue', alpha=0.9))
    plt.xlabel('Feature 1', fontsize=18)
    plt.ylabel('Feature 2', fontsize=18)

    X = [[] for _ in range(k)]
    Y = [[] for _ in range(k)]
    for j in range(k):
        for i in range(len(assign_set[j])):
            X[j].append(assign_set[j][i][0])
        for i in range(len(assign_set[j])):
            Y[j].append(assign_set[j][i][1])
        plt.scatter(X[j], Y[j], color=color_set[j], marker=mark[j], label=f'cluster {j + 1}')
        plt.scatter(center[j][0], center[j][1], c='r', marker='*', s=800)

    plt.legend()
    plt.show()


def call_(data_set, y_true):
    """
    调用sklearn包，采用k-means++进行初始化
    采用SSE与silhouette作为评价准则
    """
    # cls = KMeans(init="k-means++", n_clusters=10)
    # cls.fit(data_set)
    # cluster_labels = cls.labels_
    #
    # # 互信息分：普通互信息分、调整的互信息分、标准化互信息分
    # print(f"mutual_info_score: {mutual_info_score(y_true, cluster_labels):.2f}")
    # print(f"adjusted_mutual_info_score: {adjusted_mutual_info_score(y_true, cluster_labels):.2f}")
    # print(f"normalized_mutual_info_score: {normalized_mutual_info_score(y_true, cluster_labels):.2f}")
    # # 兰德系数
    # print(f"adjusted_rand_score: {adjusted_rand_score(y_true, cluster_labels):.2f}")

    # SSE
    sse_set = [] * 10
    for n in range(1, 11):
        cls = KMeans(n_clusters=n)
        cls.fit(data_set)
        cluster_labels = cls.labels_
        sse_set.append(cls.inertia_)

        """
        ValueError: Number of labels is 1. Valid values are 2 to n_samples - 1 (inclusive)
        """
        if n == 1:
            continue
        silhouette_avg = silhouette_score(data_set, cluster_labels)
        print(f"For n_clusters ={n} The average silhouette_score is: {silhouette_avg}  The inertia is: {cls.inertia_}")

        # plt.title('The relationship between n_clusters and SSE', fontsize=18)
        # plt.plot(range(1, 11), sse_set, color='b', marker='o')
        # plt.xlabel('n_clusters', fontsize=14)
        # plt.ylabel('SSE', fontsize=14)
        # plt.xticks(range(1, 11))
        # plt.show()

        # Visualization Result of cluster
        plt.figure(figsize=(14, 10))
        plt.title(f"Cluster of K-Means(n_cluster={n})", fontsize=25,
                  bbox=dict(facecolor='y', edgecolor='blue', alpha=0.9))
        plt.xlabel('Feature 1', fontsize=18)
        plt.ylabel('Feature 2', fontsize=18)

        # Cluster points
        for i in range(n):
            plt.scatter(data_set[cluster_labels == i, 0], data_set[cluster_labels == i, 1], color=color_set[i],
                        marker=mark[i], label=i, zorder=2)

        # Cluster centers
        plt.scatter(cls.cluster_centers_[:, 0], cls.cluster_centers_[:, 1], c='r', marker='*', s=200,
                    label='cluster center', zorder=3)

        # Cluster boundary
        x_min, x_max = data_set[:, 0].min() - 0.01, data_set[:, 0].max() + 0.01
        y_min, y_max = data_set[:, 1].min() - 0.01, data_set[:, 1].max() + 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        z = cls.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, z, zorder=1)
        plt.legend(fontsize=10)
        plt.show()


if __name__ == '__main__':
    # Step1 Loading data
    train_data_path, train_label_path = '../data/MNIST/train-images-idx3-ubyte', '../data/MNIST/train-labels-idx1-ubyte'
    data, label = data_loader(train_data_path, train_label_path, ratio=0.1)

    # Step2 Preprocessing
    data_reduction = dimension_reduction(data, mode='PCA', visualize=True)

    # Step 3 Select the best k according SSE
    # plot_sse(data_reduction, 1, 10)

    # # Step4 Cluster
    k = 10
    cluster, cluster_center = k_means(data_reduction, k)
    plot_cluster(cluster, cluster_center, k)
    # call_(data_reduction, label)

    # Step5 Evaluation (sse、silhouette)
    # print(f"SSE: {compute_sse(cluster, cluster_center)}")
