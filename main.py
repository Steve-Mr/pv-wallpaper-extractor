import time
from cluster import cluster_features
from extractor import extract_features_akaze, extract_features_akaze_speed, extract_features_orb
from untils import postprocess_speed, process_data_speed, read_images, postprocess, process_data, create_links, read_images_speed # , preprocess_images_in_parallel, cluster_images, postprocess_cluster


def main(directory):
    start_time = time.time()

    # 预处理目录下的所有图片
    image_paths, image_objects = read_images(directory)
    
    if not image_objects:
        print("No images found in the directory.")
        return
    
    # 提取特征并进行聚类
    feature_extraction_start_time = time.time()
    print(f"Reading images: {feature_extraction_start_time - start_time} seconds")

    feature_index = extract_features_akaze_speed(image_objects)
    feature_extraction_end_time = time.time()

    cluster_start_time = time.time()
    print(f"Feature extraction: {feature_extraction_end_time - feature_extraction_start_time} seconds")

    cluster_mapping = cluster_features(feature_index)
    cluster_end_time = time.time()
    print(f"Clustering: {cluster_end_time - cluster_start_time} seconds")

    
    # 对图片进行分类
    postprocess_start_time = time.time()
    image_cluster_mapping = postprocess(image_paths, feature_index, cluster_mapping)
    result = process_data(image_cluster_mapping)

    # cluster_images(image_cluster_mapping)
    postprocess_end_time = time.time()
    print(f"Postprocessing: {postprocess_end_time - postprocess_start_time} seconds")


    # print(process_data(image_cluster_mapping))

    create_links(result)
    
    # # 创建一个空字典，用于按簇索引分组图片地址
    # cluster_images_mapping = {}

    # # 将图片地址按照簇索引分组
    # for image_path, cluster_index in image_cluster_mapping.items():
    #     if cluster_index not in cluster_images_mapping:
    #         cluster_images_mapping[cluster_index] = []
    #     cluster_images_mapping[cluster_index].append(image_path)

    # # 输出每个 cluster 对应的所有图片地址
    # for cluster_index, images in cluster_images_mapping.items():
    #     print(f"Cluster {cluster_index} contains the following images:")
    #     for image_path in images:
    #         print(image_path)

    end_time = time.time()

    # 输出各阶段耗时
    print("\nTime taken for each stage:")
    print(f"Reading images: {feature_extraction_start_time - start_time} seconds")
    print(f"Feature extraction: {feature_extraction_end_time - feature_extraction_start_time} seconds")
    print(f"Clustering: {cluster_end_time - cluster_start_time} seconds")
    print(f"Postprocessing: {postprocess_end_time - postprocess_start_time} seconds")
    print(f"Total execution time: {end_time - start_time} seconds")


# 示例用法
if __name__ == "__main__":
    directory_path = "/home/maary/Videos/5th"
    main(directory_path)
