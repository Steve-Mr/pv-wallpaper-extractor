import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image
import shutil
import time
import imagehash
from picker import select_sharpest_image

def calculate_similarity(descriptor1, descriptor2):
    # 使用BFMatcher进行特征匹配
    bf = cv2.BFMatcher(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    # 应用比率测试以筛选匹配
    good_matches = []
    try:
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
            # if m.distance < 0.75 * n.distance:
                good_matches.append([m])
        # 计算相似度
        similarity = len(good_matches) / max(len(descriptor1), len(descriptor2))
    except Exception:
        similarity = 1

    print(similarity)
    return similarity


def extract_features(img_path):
    # print(img_path)
    # 打开图片
    img = Image.open(img_path)
    
    # 获取原始图片尺寸
    width, height = img.size
    
    # 将图片横边缩放到原来的十分之一
    new_width = int(width / 5)
    # 根据比例计算新的高度
    new_height = int(height * (new_width / width))
    
    # 缩放图片
    resized_img = img.resize((new_width, new_height))

    np_img = np.array(resized_img)
    # 初始化特征提取器
    akaze = cv2.AKAZE_create()

    # 检测特征点和计算描述子
    keypoints, descriptors = akaze.detectAndCompute(np_img, None)

    # # 可视化特征点并保存结果
    # # img_with_keypoints = cv2.drawKeypoints(img, keypoints, img, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # for kp in keypoints:
    #     x, y = kp.pt
    #     cv2.circle(np_img, (int(x), int(y)), 1, (0, 255, 0), -1)  # Draw a green circle at the keypoint position

    # # 保存可视化结果
    # save_path = os.path.join('feature_visualizations', f"keypoints_img_{os.path.basename(img_path)}.jpg")
    # # cv2.imwrite(save_path, img_with_keypoints)
    # cv2.imwrite(save_path, np_img)

    return descriptors


def extract_hash(image_path, hash_size=4):
    img = Image.open(image_path)

    # # 获取原始图片尺寸
    # width, height = img.size
    
    # # 将图片横边缩放到原来的十分之一
    # new_width = int(width / 20)
    # # 根据比例计算新的高度
    # new_height = int(height * (new_width / width))
    
    # 缩放图片
    image = img.resize((16, 16))
    hash = imagehash.phash(image, hash_size=20)
    return hash


def calculate_hash_similarity(hash1, hash2):

    # 计算相似度
    similarity = 1 - abs(hash1-hash2)/len(hash1)
    # print(similarity)
    return similarity


def process_images(images):
    """
    处理一组图片
    """
    window_size = 10
    group_results = defaultdict(list)
    window_start = 0
    window_end = window_size

    while(images):
        # match_image = images[0]
        match_features = extract_features(images[0])
        if match_features is not None and len(match_features) > 5: break
        else:
            print(images[0])
            del images[0]    

    if not images:
        return group_results
    
    # 第一个图片作为匹配符
    match_image = images[0]
    match_features = extract_features(match_image)    
    group_results[match_image].append(match_image)
    
    for image in islice(images, 1, None):
        features = extract_features(image)
        if features is None or len(features) < 5: 
            print(image)
            continue
        similarity = calculate_similarity(match_features, features)
        # print(similarity)
        
        # 若相似度较高且与前面元素相似度类似，则认为同一类图片
        if similarity > 0.7:  # 示例相似度阈值
            group_results[match_image].append(image)
        else:
            # 将窗口开始端移动到该元素，末尾端同步移动
            match_image = image
            match_features = features
            window_start += 1
            group_results[match_image].append(match_image)

        # 移动窗口末尾
        window_end += 1

    # print(group_results)
    return group_results

def merge_results(results):
    """
    合并处理结果并按照 key 大小排序
    """
    merged_results = []

    for group_result in results:
        for key, value in group_result.items():
            merged_results.append({key: value})

    merged_results = sorted(merged_results, key=lambda x: os.path.basename(list(x.keys())[0]))

    return merged_results


def process_images_in_thread(images, thread_id):
    """
    多线程处理图片
    """
    print(f"Thread {thread_id} processing {len(images)} images")
    group_results = process_images(images)
    print(f"Thread {thread_id} finished processing")
    return group_results

def merge_groups(groups):
    """
    合并图片组
    """
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

            if calculate_hash_similarity(extract_hash(current_images[-1]), extract_hash(next_images[0])) > 0.7:
                merged_group.update(next_group)
                groups[j] = None
                j += 1
            else:
                break

        merged_groups.append(merged_group)
        i = j

    return merged_groups


def copy_sharpest_images(output_dir):
    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历 input_dir 下的所有子目录
    for root, dirs, files in os.walk(output_dir):
        # 创建线程池
        with ThreadPoolExecutor() as executor:
            # 使用多线程处理每个目录中的图片
            for directory in dirs:
                dir_path = os.path.join(root, directory)
                # 在线程池中异步执行任务
                executor.submit(process_directory, dir_path, output_dir)

def process_directory(directory, output_dir):
    # 使用 select_sharpest_image 方法选取最清晰的图片
    sharpest_image = select_sharpest_image(directory)
    if sharpest_image:
        # 获取最清晰图片的文件名
        filename = os.path.basename(sharpest_image)
        # 构造目标路径
        destination_path = os.path.join(output_dir, filename)
        # 复制图片到 output_dir
        shutil.copy(sharpest_image, destination_path)
        print(f"Copied {sharpest_image} to {destination_path}")

def copy_images(results, output_dir='output'):
    """
    根据结果将图片复制到输出文件夹中
    """
    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历结果，每个结果对应一个文件夹
    for idx, group_result in enumerate(results):
        folder_path = os.path.join(output_dir, str(idx + 1))  # 输出文件夹路径
        os.makedirs(folder_path)  # 创建输出文件夹

        # 遍历当前结果中的地址列表，并将图片复制到对应文件夹中
        for image_list in group_result.values():
            # sharpest_path = select_sharpest_image(image_list)
            # image_filename = os.path.basename(sharpest_path)
            # # 构造目标路径
            # destination_path = os.path.join(output_dir, image_filename)
            # shutil.copy(sharpest_path, destination_path)
            for image_path in image_list:
                # 获取图片文件名
                image_filename = os.path.basename(image_path)
                # 构造目标路径
                destination_path = os.path.join(folder_path, image_filename)
                # 复制图片
                shutil.copy(image_path, destination_path)
    copy_sharpest_images(output_dir)


def main(directory):
    start_total_time = time.time()

    # 获取目录下所有图片文件
    start_time = time.time()
    images = sorted([os.path.join(directory, filename) for filename in os.listdir(directory)])
    end_time = time.time()
    print(f"获取图片文件耗时: {end_time - start_time:.2f} 秒")

    # 获取系统核心数
    num_cores = multiprocessing.cpu_count()
    # num_threads = max(1, int(3 * num_cores / 4))
    num_threads = num_cores

    # 将图片均匀分组，每组数量为图片总数除以线程数
    start_time = time.time()
    chunk_size = len(images) // num_threads
    image_groups = [images[i:i+chunk_size] for i in range(0, len(images), chunk_size)]
    end_time = time.time()
    print(f"分组耗时: {end_time - start_time:.2f} 秒")

    # 多线程处理图片
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_images_in_thread, image_groups, range(num_threads)))
    end_time = time.time()
    print(f"多线程处理图片耗时: {end_time - start_time:.2f} 秒")

    # 合并处理结果
    start_time = time.time()
    merged_results = merge_results(results)
    end_time = time.time()
    print(f"合并处理结果耗时: {end_time - start_time:.2f} 秒")

    # 合并图片组
    start_time = time.time()
    merged_groups = merge_groups(merged_results)
    end_time = time.time()
    print(f"合并图片组耗时: {end_time - start_time:.2f} 秒")

    # 复制图片
    start_time = time.time()
    copy_images(merged_groups)
    end_time = time.time()
    print(f"复制图片耗时: {end_time - start_time:.2f} 秒")

    end_total_time = time.time()
    print(f"总耗时: {end_total_time - start_total_time:.2f} 秒")

# 测试示例
directory_path = "/home/maary/Videos/baop2"
result = main(directory_path)
