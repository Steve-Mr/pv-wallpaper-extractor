import os
import cv2
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from picker import select_sharpest_image  # 导入选择最清晰图片的函数
import imagehash
import shutil
import time
from PIL import Image
import numpy as np

# def calculate_image_sharpness(image):
#     return cv2.Laplacian(image, cv2.CV_64F).var()

def calculate_similarity(hash1, hash2):

    # 计算相似度
    similarity = 1 - abs(hash1-hash2)/len(hash1)
    # print(similarity)
    return similarity

def extract_features(image_path, hash_size=16):
    # img = cv2.imread(image_path)
    img = Image.open(image_path)
    img = img.resize((8, 8))  # 缩小图片尺寸以加快处理速度
    # img = np.array(img)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    hash = imagehash.phash(img, hash_size=hash_size)
    return hash

def process_images(images):
    group_results = defaultdict(list)

    match_image = None
    match_features = None

    for image in images:
        features = extract_features(image)
        if features is None or len(features) < 5:
            continue
        if match_image is None:
            match_image = image
            match_features = features
            group_results[match_image].append(match_image)
            continue

        similarity = calculate_similarity(match_features, features)
        if similarity > 0.8:  # 示例相似度阈值
            group_results[match_image].append(image)
        else:
            match_image = image
            match_features = features
            group_results[match_image].append(match_image)

    return group_results

def merge_results(results):
    merged_results = []

    for group_result in results:
        for key, value in group_result.items():
            merged_results.append({key: value})

    merged_results = sorted(merged_results, key=lambda x: os.path.basename(list(x.keys())[0]))

    return merged_results

def merge_groups(groups):
    merged_groups = []

    i = 0
    while i < len(groups):
        current_group = groups[i]

        if current_group is None:
            i += 1
            continue

        merged_group = current_group.copy()
        j = i + 1

        while j < len(groups) and groups[j]:
            next_group = groups[j]
            current_images = list(merged_group.keys())
            next_images = list(next_group.keys())

            if calculate_similarity(extract_features(current_images[-1]), extract_features(next_images[0])) > 0.8:
                merged_group.update(next_group)
                groups[j] = None
                j += 1
            else:
                break

        merged_groups.append(merged_group)
        i = j

    return merged_groups

def copy_images(results, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, group_result in enumerate(results):
        folder_path = os.path.join(output_dir, str(idx + 1))
        os.makedirs(folder_path)

        for image_list in group_result.values():
            for image_path in image_list:
                image_filename = os.path.basename(image_path)
                destination_path = os.path.join(folder_path, image_filename)
                shutil.copy(image_path, destination_path)

def main(directory):
    start_total_time = time.time()

    images = sorted([os.path.join(directory, filename) for filename in os.listdir(directory)])

    num_cores = os.cpu_count()
    num_threads = num_cores

    chunk_size = len(images) // num_threads
    image_groups = [images[i:i+chunk_size] for i in range(0, len(images), chunk_size)]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_images, image_groups))

    merged_results = merge_results(results)
    merged_groups = merge_groups(merged_results)

    copy_images(merged_groups)

    end_total_time = time.time()
    print(f"总耗时: {end_total_time - start_total_time:.2f} 秒")

if __name__ == "__main__":
    main("/home/maary/Videos/debug")
