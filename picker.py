import cv2
import os

def calculate_image_sharpness(image):
    # 计算Laplacian算子的方差来评估图像的清晰度
    return cv2.Laplacian(image, cv2.CV_64F).var()

# def select_sharpest_image(images):
#     max_sharpness = 0
#     sharpest_image = None
    
#     # 遍历目录中的所有图片
#     # for filepath in os.listdir(images):
#     for filepath in images:
#         if filepath.endswith(".jpg") or filepath.endswith(".png") or filepath.endswith('.webp'):
#             # filepath = os.path.join(directory, filename)
#             img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # 以灰度图像读取，加快处理速度
#             if img is not None:
#                 sharpness = calculate_image_sharpness(img)
#                 if sharpness > max_sharpness:
#                     max_sharpness = sharpness
#                     # sharpest_image = cv2.imread(filepath)  # 以彩色图像读取
#                     # 如果要求只返回路径，则返回 filepath

#     return filepath

def select_sharpest_image(directory):
    max_sharpness = 0
    sharpest_image = None
    
    # 遍历目录中的所有图片
    for filename in os.listdir(directory):
    # for filename in images:
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith('.webp'):
            filepath = os.path.join(directory, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # 以灰度图像读取，加快处理速度
            if img is not None:
                sharpness = calculate_image_sharpness(img)
                if sharpness > max_sharpness:
                    max_sharpness = sharpness
                    # sharpest_image = cv2.imread(filepath)  # 以彩色图像读取
                    # 如果要求只返回路径，则返回 filepath

    return filepath

# # 调用函数并传入目录地址
# sharpest_img = select_sharpest_image("path_to_directory")

# # 显示清晰度最高的图片
# if sharpest_img is not None:
#     cv2.imshow("Sharpest Image", sharpest_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("未找到图片")
