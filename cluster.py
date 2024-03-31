import hdbscan
# import fast_hdbscan
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
# from dbscan import DBSCAN
from sklearn.cluster import DBSCAN

def cluster_features(feature_index):
    # 将特征描述子和对应的索引提取出来
    descriptors = []
    indices = []
    for key, value in feature_index.items():
        indices.append(key)
        descriptors.append(value)
    descriptors = np.array(descriptors)

    print('culster started')

    # # 使用 HDBSCAN 进行聚类
    # # clusterer = hdbscan.HDBSCAN(min_cluster_size=35, min_samples=15)
    # clusterer = fast_hdbscan.HDBSCAN(min_cluster_size=40, min_samples=10)
    # labels = clusterer.fit_predict(descriptors)

    # 使用 DBSCAN 进行聚类
    # clusterer = DBSCAN(eps=200, min_samples=5)
    clusterer = DBSCAN(eps=150, min_samples=2)
    labels = clusterer.fit_predict(descriptors)


    # 假设 labels 是聚类算法输出的聚类标签
    # descriptors 是特征描述子
    # 这里假设 descriptors 是一个二维数组，每一行代表一个特征点的描述子

    # 将特征点按照聚类标签分组
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(descriptors[idx])

    print('cluster finished')

    # 构建簇与特征索引的对应关系
    cluster_mapping = {}
    for i, label in enumerate(labels):
        if label != -1:  # -1 表示噪声点
            cluster_mapping.setdefault(label, []).append(indices[i])

    return cluster_mapping
