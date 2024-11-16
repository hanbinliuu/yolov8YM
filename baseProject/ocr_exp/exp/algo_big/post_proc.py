import cv2
import numpy as np
from PIL import Image, ImageEnhance


def denoise_image(input_image):
    # 将输入图像转换为NumPy数组
    input_image = np.array(input_image)
    # 使用高斯滤波进行图像降噪
    denoised_image = cv2.GaussianBlur(input_image, (5, 5), 0)
    # 将NumPy数组转换回PIL图像
    denoised_image = Image.fromarray(denoised_image)
    return denoised_image


file_path = '/Users/hanbinliu/Desktop/test_rectangle_big.png'
input_image = Image.open(file_path)

# 设置对比度增强因子（1.0 表示原始对比度，大于 1.0 表示增强对比度，小于 1.0 表示减小对比度）
contrast_factor = 0.9
# 对比度增强
enhancer = ImageEnhance.Contrast(input_image)
output_image = enhancer.enhance(contrast_factor)

# 图像降噪
denoised_image = denoise_image(output_image)

# 保存降噪后的图像
output_image.save("/Users/hanbinliu/Desktop/denoised_output_big.png")

temp = 1
