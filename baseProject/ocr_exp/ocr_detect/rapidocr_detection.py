from rapidocr_onnxruntime import RapidOCR
import cv2
import numpy as np

engine = RapidOCR()
img_path = r"C:\Users\Administrator\Desktop\denoised.png"

# ROI 抠图
# img = cv2.imread(img_path)
# cv2.namedWindow("image", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)  # 是否显示网格
# rect = cv2.selectROI("image", img, False, False)
# (x, y, w, h) = rect
# image = img[y: y + h, x: x + w]

# # 高斯模糊
# k_size = (3, 3)
# image_blur = cv2.GaussianBlur(image, k_size, 0)
# kernel = np.ones((3, 3))
# image_dilated = cv2.dilate(image_blur, kernel, iterations=1)
#
# # 局部二值化
# image_gray = cv2.cvtColor(image_dilated, cv2.COLOR_BGR2GRAY)
# adaptive_threshold = cv2.adaptiveThreshold(image_gray, 225, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
#
# # 全局二值化
# # threshold_value = 150
# # max_value = 255
# # ret, binary_image = cv2.threshold(image_blur, threshold_value, max_value, cv2.THRESH_BINARY)

# cv2.imshow('Binary Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

result, elapse = engine(img_path)
print(result)