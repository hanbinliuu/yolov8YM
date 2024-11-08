import cv2
import numpy as np


def find_center_and_intersection(x1, y1, x2, y2):
    # 计算中心点的坐标
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    intersection_point = (center_x, 0)

    # 返回中心点坐标和与 x 轴的交点坐标
    return (center_x, center_y), intersection_point


def img_process(img):
    image_blur = cv2.GaussianBlur(img, (5, 5), 1)
    gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 轮廓提取
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# img = cv2.imread(r"C:\Users\lhb\Desktop\test\Image_20240103170917810.bmp")
img = cv2.imread(r"C:\Users\lhb\Desktop\test2\Image_20240108095047163.bmp")
x1, y1, w1, h1 = 1925, 2079, 86, 236
# x1,y1,w1,h1 = 2350, 811, 75, 257
# x,y,w,h =2839,1170,70,436
# x,y,w,h =3077, 2172, 60, 327
imCrop = img[y1: y1 + h1, x1: x1 + w1]

# 二值化处理
image_blur = cv2.GaussianBlur(imCrop, (5, 5), 1)
gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 轮廓提取
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
for i in range(len(contours)):
    if cv2.contourArea(contours[i]) > cv2.contourArea(cnt):
        cnt = contours[i]

# 过滤掉x坐标0的点
filtered_points = []
for point in cnt:
    x, y = point[0]  # 提取点的坐标
    if x > 0:
        filtered_points.append(point)
filtered_cnt = np.array([filtered_points], dtype=np.int32)

leftmost_point = None
min_x = float('inf')
for point in filtered_cnt[0]:
    x, y = point[0]  # Access the coordinates using indexing
    if x < min_x:
        min_x = x
        leftmost_point = (x, y)

rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
print("rightmost[0] - leftmost_point[0] = {}".format(rightmost[0] - leftmost_point[0]))

# 映射回原图上的坐标点
original_leftmost_point_x = x1 + leftmost_point[0]
original_leftmost_point_y = y1 + leftmost_point[1]
original_rightmost_point_x = x1 + rightmost[0]
original_rightmost_point_y = y1 + rightmost[1]
origin_left = (original_leftmost_point_x, original_leftmost_point_y)
origin_right = (original_rightmost_point_x, original_rightmost_point_y)

mapped_cnt = []
for point in filtered_cnt[0]:
    x, y = point[0]
    mapped_cnt.append((x + x1, y + y1))
mapped_cnt = np.array([mapped_cnt], dtype=np.int32)

cv2.drawContours(img, mapped_cnt, -1, (0, 0, 255), 2)
cv2.circle(img, origin_left, 5, (0, 255, 0), -1)
cv2.circle(img, origin_right, 5, (0, 255, 0), -1)

# 保存
cv2.imwrite(r"C:\Users\lhb\Desktop\image_blur5.bmp", img)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
