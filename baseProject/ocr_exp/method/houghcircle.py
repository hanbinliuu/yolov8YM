import cv2
import numpy as np

img = cv2.imread('/Users/hanbinliu/Desktop/testing.png', cv2.IMREAD_COLOR)
# img = cv2.imread('/Users/hanbinliu/Desktop/ocr检测/ocr/test4.png', cv2.IMREAD_COLOR)

# 转换图像为灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯模糊，以减少噪音
gray = cv2.GaussianBlur(gray, (1, 1), 2)

# 霍夫圆变换
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=100,
    param1=200,
    param2=200,
    minRadius=100,
    maxRadius=0
)


# 寻找最近的圆
if circles is not None:
    circles = np.uint16(np.around(circles))
    image_center = (img.shape[1] // 2, img.shape[0] // 2)  # 图像中心点

    closest_circle = None
    min_distance = float('inf')

    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        radius = circle[2]

        # 计算中心点与图像中心点的距离
        distance = np.sqrt((center[0] - image_center[0])**2 + (center[1] - image_center[1])**2)

        # 如果这个圆更接近，则更新最近的圆
        if distance < min_distance:
            min_distance = distance
            closest_circle = (center, radius)

    if closest_circle is not None:
        center, radius = closest_circle
        cv2.circle(img, center, radius, (0, 0, 255), 2)  # 绘制最近的圆

# 显示图像
cv2.imshow('Detected Circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
