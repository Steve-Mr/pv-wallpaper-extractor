import cv2
import os
import concurrent.futures

def extract_features_orb(images, save_dir="feature_visualizations"):
    # 初始化特征提取器
    orb = cv2.ORB_create(nfeatures=300)

    # 存储特征点的索引
    feature_index = {}

    # 检查保存目录是否存在，如果不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 遍历每张图片
    for idx, img in enumerate(images):
        # 转换为灰度图像
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 检测特征点和计算描述子
        keypoints, descriptors = orb.detectAndCompute(img, None)

        num_keypoints = len(keypoints)

        # 可视化特征点并保存结果
        # img_with_keypoints = cv2.drawKeypoints(img, keypoints, img, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        for kp in keypoints:
            x, y = kp.pt
            cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)  # Draw a green circle at the keypoint position

        # 保存可视化结果
        save_path = os.path.join(save_dir, f"keypoints_img_{idx}.jpg")
        # cv2.imwrite(save_path, img_with_keypoints)
        cv2.imwrite(save_path, img)
        
        # 将特征点和对应的描述子添加到索引中
        for i, keypoint in enumerate(keypoints):
            feature_index[(idx, i)] = descriptors[i]

        print(f"Extracted features from image {idx} and saved visualization to {save_path}. Num keypoints: {num_keypoints}")
    
    return feature_index


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

        # num_keypoints = len(keypoints)

        # # 可视化特征点并保存结果
        # # img_with_keypoints = cv2.drawKeypoints(img, keypoints, img, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
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





def extract_features_akaze_speed(images, save_dir="feature_visualizations"):
    # 初始化特征提取器
    akaze = cv2.AKAZE_create()

    # 存储特征点的索引
    feature_index = {}

    # 检查保存目录是否存在，如果不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 并行处理每张图片
    def process_image(idx, img):
        nonlocal feature_index
        # 检测特征点和计算描述子
        keypoints, descriptors = akaze.detectAndCompute(img, None)

        # 将特征点和对应的描述子添加到索引中
        for i, keypoint in enumerate(keypoints):
            feature_index[(idx, i)] = descriptors[i]

    # 创建线程池
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交任务并等待完成
        futures = [executor.submit(process_image, idx, img) for idx, img in enumerate(images)]
        concurrent.futures.wait(futures)

    return feature_index
