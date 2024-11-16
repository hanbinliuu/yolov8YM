import pytesseract
import cv2
import numpy as np
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r"E:/software/Tesseract/tesseract.exe"
from PIL import Image


if __name__ == '__main__':

    file_path = 'C:/Users/Administrator/Desktop/20231103-132609.jpg'
    image = cv2.imread(file_path)

    # # 转为灰度图
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # 使用自适应阈值二值化
    # binary_image = cv2.adaptiveThreshold(
    #     gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    # )
    #
    # cv2.imshow('Adaptive Binary Image', binary_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    x = 2106
    y = 1605
    w = 105
    h = 102
    image = image[y: y + h, x: x + w]

    # 高斯模糊
    k_size = (3, 3)
    image_blur = cv2.GaussianBlur(image, k_size, 0)
    kernel = np.ones((3, 3), np.uint8)
    image_dilated = cv2.dilate(image_blur, kernel, iterations=1)

    # 局部二值化
    image_gray = cv2.cvtColor(image_dilated, cv2.COLOR_BGR2GRAY)
    adaptive_threshold = cv2.adaptiveThreshold(image_gray, 225, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

    # 全局二值化
    # threshold_value = 150
    # max_value = 255
    # ret, binary_image = cv2.threshold(image_blur, threshold_value, max_value, cv2.THRESH_BINARY)

    # cv2.imshow('Binary Image', adaptive_threshold)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    text = pytesseract.image_to_string(Image.fromarray(adaptive_threshold))

    print(text)
