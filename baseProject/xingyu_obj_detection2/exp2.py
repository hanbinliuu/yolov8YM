import cv2
import math
import numpy as np


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_closest_points(x1, y1, x2, y2, x3, y3, x4, y4):
    points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    min_distance = float('inf')
    closest_points = None

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = calculate_distance(points[i][0], points[i][1], points[j][0], points[j][1])
            if distance < min_distance:
                min_distance = distance
                closest_points = (points[i], points[j])

    diff_x = abs(closest_points[0][0] - closest_points[1][0])
    return diff_x


def find_center_and_intersection(x1, y1, x2, y2):
    # 计算中心点的坐标
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    intersection_point = (center_x, 0)
    # 返回中心点坐标和与 x 轴的交点坐标
    return (center_x, center_y), intersection_point


def calculate_angle(x1, y1, x2, y2, x3, y3, x4, y4):
    """ 计算两条线段之间的夹角 """
    # Vectors
    v1 = [x2 - x1, y2 - y1]
    v2 = [x4 - x3, y4 - y3]

    # Dot product
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]

    # Magnitudes
    magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    # Calculate angle in radians
    angle_radians = math.acos(dot_product / (magnitude_v1 * magnitude_v2))

    # Convert angle to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


def img_process(img):
    image_blur = cv2.GaussianBlur(img, (5, 5), 1)
    gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 轮廓提取
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 选取最大轮廓
    cnt = contours[0]
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > cv2.contourArea(cnt):
            cnt = contours[i]

    # 可视化
    cv2.drawContours(img, cnt, -1, (0, 0, 255), 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cnt


def find_bound_point(cnt):
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
    return leftmost_point, rightmost, filtered_cnt


def map_original_coordinate(left, right, filtered_cnt, x1, y1):
    # 映射回原图上的坐标点
    original_leftmost_point_x = x1 + left[0]
    original_leftmost_point_y = y1 + left[1]
    original_rightmost_point_x = x1 + right[0]
    original_rightmost_point_y = y1 + right[1]
    origin_left = (original_leftmost_point_x, original_leftmost_point_y)
    origin_right = (original_rightmost_point_x, original_rightmost_point_y)

    mapped_cnt = []
    for point in filtered_cnt[0]:
        x, y = point[0]
        mapped_cnt.append((x + x1, y + y1))
    mapped_cnt = np.array([mapped_cnt], dtype=np.int32)

    return mapped_cnt, origin_left, origin_right


if __name__ == '__main__':

    # 25mm镜头
    img = cv2.imread(r"C:\Users\lhb\Desktop\test2\Image_20240108095047163.bmp")
    x1, y1, w1, h1 = 2611, 2605, 75, 65
    x2, y2, w2, h2 = 2652, 2686, 71, 106

    # 50mm镜头
    # img = cv2.imread(r"C:\Users\baseProject\Desktop\test2\Image_20240108095447745.bmp")
    # x1, y1, w1, h1 = 2119, 2715, 100, 77
    # x2, y2, w2, h2 = 2171, 2812, 78, 102

    imCrop = img[y1: y1 + h1, x1: x1 + w1]
    imCrop2 = img[y2: y2 + h2, x2: x2 + w2]

    max_contours = img_process(imCrop)
    max_countours2 = img_process(imCrop2)

    leftmost, rightmost, filtered_cnt = find_bound_point(max_contours)
    leftmost2, rightmost2, filtered_cnt2 = find_bound_point(max_countours2)

    mapped_cnt, origin_left, origin_right = map_original_coordinate(leftmost, rightmost, filtered_cnt, x1, y1)
    mapped_cnt2, origin_left2, origin_right2 = map_original_coordinate(leftmost2, rightmost2, filtered_cnt2, x2, y2)

    cv2.drawContours(img, mapped_cnt, -1, (0, 0, 255), 2)
    cv2.circle(img, origin_left, 2, (   0, 255, 0), -1)
    cv2.circle(img, origin_right, 2, (0, 255, 0), -1)

    cv2.drawContours(img, mapped_cnt2, -1, (0, 255, 0), 2)  # Different color for clarity
    cv2.circle(img, origin_left2, 2, (0, 0, 255), -1)  # Different color for clarity
    cv2.circle(img, origin_right2, 2, (0, 0, 255), -1)  # Different color for clarity

    cv2.imwrite(r"C:\Users\lhb\Desktop\test2\combined_image2.bmp", img)
    diff_x = find_closest_points(origin_left[0], origin_left[1], origin_right[0], origin_right[1],
                                 origin_left2[0], origin_left2[1], origin_right2[0], origin_right2[1])

    temp = 1
