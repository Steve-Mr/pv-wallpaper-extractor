from PIL import Image
import os 
import concurrent.futures
import numpy as np
import cv2
import hdbscan
import fast_hdbscan
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

def read_images(directory):
    # 初始化图片对象列表
    image_paths = []
    image_objects = []

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif','.bmp', 'webp')):
            image_paths.append(filepath)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        image_objects = list(executor.map(preprocess, image_paths))

    return image_paths, image_objects


def preprocess(image_path):
    # 打开图片
    img = Image.open(image_path)
    
    # 获取原始图片尺寸
    width, height = img.size
    
    # 将图片横边缩放到原来的十分之一
    new_width = int(width / 10)
    # 根据比例计算新的高度
    new_height = int(height * (new_width / width))
    
    # 缩放图片
    resized_img = img.resize((new_width, new_height))
    
    # 将 PIL 图片对象转换为 NumPy 数组
    np_img = np.array(resized_img)
    
    # 将彩色图像转换为灰度图像
    # gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    
    # 返回处理后的灰度图像
    # return gray
    return np_img


def extract_features_akaze(images, save_dir="feature_visualizations"):
    # 初始化特征提取器
    akaze = cv2.AKAZE_create()

    # 存储特征点的索引
    feature_index = {}

    # 检查保存目录是否存在，如果不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 遍历每张图片
    for idx, img in enumerate(images):
        
        # 检测特征点和计算描述子
        keypoints, descriptors = akaze.detectAndCompute(img, None)

        num_keypoints = len(keypoints)

        # # 可视化特征点并保存结果
        # for kp in keypoints:
        #     x, y = kp.pt
        #     cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)  # Draw a green circle at the keypoint position

        # # 保存可视化结果
        # save_path = os.path.join(save_dir, f"keypoints_img_{idx}.jpg")
        # # cv2.imwrite(save_path, img_with_keypoints)
        # cv2.imwrite(save_path, img)
        
        # 将特征点和对应的描述子添加到索引中
        for i, keypoint in enumerate(keypoints):
            feature_index[(idx, i)] = descriptors[i]

        # print(f"Extracted features from image {idx} and saved visualization to {save_path}. Num keypoints: {num_keypoints}")
    
    return feature_index


def cluster_features(feature_index):
    # 将特征描述子和对应的索引提取出来
    descriptors = []
    indices = []
    for key, value in feature_index.items():
        indices.append(key)
        descriptors.append(value)
    descriptors = np.array(descriptors)

    print('culster started')

    # 使用 HDBSCAN 进行聚类
    # clusterer = hdbscan.HDBSCAN(min_cluster_size=35, min_samples=15)
    clusterer = fast_hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10)
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

def classify_images(image_paths, cluster_mapping):
    image_classes = {}

    # 根据聚类结果对图片进行分类
    for cluster_label, indices in cluster_mapping.items():
        # 获取属于当前簇的图片索引
        cluster_image_paths = [image_paths[idx[0]] for idx in indices]
        # 将图片索引添加到对应的类别中
        image_classes[cluster_label] = cluster_image_paths

    return image_classes

# 读取图片
image_directory = "/home/maary/Videos/test"
image_paths, image_objects = read_images(image_directory)

# 提取特征并聚类
feature_index = extract_features_akaze(image_objects)
cluster_mapping = cluster_features(feature_index)

# 根据聚类结果对图片进行分类
image_classes = classify_images(image_paths, cluster_mapping)

# 输出每个类别中的图片路径
for cluster_label, images in image_classes.items():
    print(f"Cluster {cluster_label}:")
    for image_path in images:
        print(image_path)
