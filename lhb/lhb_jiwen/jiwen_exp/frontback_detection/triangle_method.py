import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt


def distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def zmMinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    return cv.erode(src, np.ones((2 * r + 1, 2 * r + 1)))


def dark_channel_catch_hole(path):
    image = cv.imread(path)
    # 高斯模糊
    image_gauss = cv.GaussianBlur(image, (5, 5), 0)
    # 暗通道
    image_dark = np.min(image_gauss, 2)
    Dark_Channel = zmMinFilterGray(image_dark, 7)
    # 二值化，第二个参数是阈值
    (_, thresh) = cv.threshold(Dark_Channel, 90, 255, cv.THRESH_BINARY)
    # 腐蚀
    thresh = cv.dilate(thresh, None, iterations=3)
    # cv.imshow("thresh", thresh)
    cnts, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(thresh, cnts, -1, (255), -1)
    image_copy = image.copy()
    centerx = []
    centery = []
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        # 框体限定
        # if w < 180 or h < 180:
        if w < 150 or h < 150:
            continue
        if w > 500 or h > 500:
            continue
        cv.rectangle(image_copy, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 0, 255), 10)  # 画框

        center_x = x + w // 2
        center_y = y + h // 2
        centerx.append(center_x)
        centery.append(center_y)

    # cv.imshow("image_copy", image_copy)
    # cv.waitKey(0)

    return centerx, centery


def find_closest_points(points):
    n = len(points)
    distances = {}

    for i in range(n):
        for j in range(i + 1, n):
            dist = distance(points[i], points[j])
            distances[(i, j)] = dist

    return distances


def find_idx(distances):
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    nearest_three = sorted_distances[:3]
    farest_two = sorted_distances[-1]
    nearest_indices = [item[0] for item in nearest_three]
    farthest_indices = [farest_two[0]]

    flattened_indices = set([index for pair in nearest_indices for index in pair])  # 将元组索引展开成一维列表
    flattened_indices2 = set([index for pair in farthest_indices for index in pair])
    # 找交集
    return flattened_indices.intersection(flattened_indices2)


def find_longest_shortest_side(vertices):
    # vertices是三个顶点的列表，每个顶点是一个元组 (x, y)

    A, B, C = vertices

    # 计算三个边的长度和对应的点
    AB = distance(A, B)
    BC = distance(B, C)
    CA = distance(C, A)

    sides = [(AB, A, B), (BC, B, C), (CA, C, A)]

    # 找到最长边和最短边
    longest_side, longest_start, longest_end = max(sides, key=lambda x: x[0])
    shortest_side, shortest_start, shortest_end = min(sides, key=lambda x: x[0])

    return (longest_start, longest_end), (shortest_start, shortest_end)


def find_shortest_edge(points):
    # 计算三角形三条边的长度
    A, B, C = points
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)

    # 找到最短的边的长度
    shortest_edge_length = min(a, b, c)

    # 返回最短边的端点
    if shortest_edge_length == a:
        return B, C
    elif shortest_edge_length == b:
        return A, C
    else:
        return A, B

def rotataion_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def rotate_to_horizontal(points, plot=True):
    # 找到最短边的两个顶点
    shortest_vertex, other_vertex = find_shortest_edge(points)

    # 计算最短边的角度
    angle = np.arctan2(other_vertex[1] - shortest_vertex[1], other_vertex[0] - shortest_vertex[0])

    # 计算旋转矩阵
    rotation_matrix = rotataion_matrix(angle)

    # 进行旋转
    rotated_points = np.dot(points, rotation_matrix)
    longest, shortest = find_longest_shortest_side(rotated_points)

    y_longside_max = max(longest[0][1], longest[1][1])
    y_shortside_max = shortest[0][1]

    if y_longside_max == y_shortside_max:
        angle = angle + np.pi # 旋转180度
        rotation_matrix = rotataion_matrix(angle)
        rotated_points = np.dot(points, rotation_matrix)

    if plot:
        # plot_triangle(points, 'Original')
        plot_triangle(rotated_points, 'Rotated to Horizontal')


    return rotated_points


def plot_triangle(points, title):
    plt.figure()
    plt.plot(points[:, 0], points[:, 1], 'bo-')
    plt.plot([points[0, 0], points[-1, 0]], [points[0, 1], points[-1, 1]], 'bo-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()


# 判断在左侧还是右侧
def calculate_angle(side_a, side_b, side_c):
    return math.degrees(math.acos((side_b ** 2 + side_c ** 2 - side_a ** 2) / (2 * side_b * side_c)))


def find_min_and_obtuse_points(points_list):
    point1 = points_list[0]
    point2 = points_list[1]
    point3 = points_list[2]

    # Calculate side lengths using distance formula
    side_a = distance(point2, point3)
    side_b = distance(point3, point1)
    side_c = distance(point1, point2)

    angle_A = calculate_angle(side_a, side_b, side_c)
    angle_B = calculate_angle(side_b, side_c, side_a)
    angle_C = 180 - angle_A - angle_B

    min_angle_vertex = None
    if angle_A < angle_B and angle_A < angle_C:
        min_angle_vertex = point1
    elif angle_B < angle_A and angle_B < angle_C:
        min_angle_vertex = point2
    else:
        min_angle_vertex = point3

    obtuse_angle_vertex = None
    if angle_A > 90:
        obtuse_angle_vertex = point1
    elif angle_B > 90:
        obtuse_angle_vertex = point2
    else:
        obtuse_angle_vertex = point3

    return min_angle_vertex, obtuse_angle_vertex


def is_right_oriented(min_angle_vertex, obtuse_angle_vertex):
    return min_angle_vertex[0] > obtuse_angle_vertex[0]


def triangle_selection(image_path):
    centerx, centery = dark_channel_catch_hole(image_path)
    zip_data = np.float32([[x, y] for x, y in zip(centerx, centery)])
    distances = find_closest_points(zip_data)
    removal_idx = list(find_idx(distances))[0]

    centerx.pop(removal_idx)
    centery.pop(removal_idx)
    coordinates = np.float32([[x, y] for x, y in zip(centerx, centery)])
    return coordinates

def check_flip(template_path, test_path):
    template_cord = triangle_selection(template_path)
    test_cord = triangle_selection(test_path)

    rotated_points = rotate_to_horizontal(template_cord)
    rotated_points2 = rotate_to_horizontal(test_cord)

    min_angle_vertex, obtuse_angle_vertex = find_min_and_obtuse_points(rotated_points2)
    min_angle_vertex2, obtuse_angle_vertex2 = find_min_and_obtuse_points(rotated_points)

    if is_right_oriented(min_angle_vertex, obtuse_angle_vertex) == is_right_oriented(min_angle_vertex2, obtuse_angle_vertex2):
        return "not flip"
    else:
        return "flip"


if __name__ == '__main__':

    # test pic
    # 53 69 70 71
    # 53 69 not flip
    # 69 70 not flip
    # 69 71 flip
    # 70 71 flip

    template_path = "/Users/hanbinliu/Downloads/IMG_1384.jpeg"
    test_path = "/Users/hanbinliu/Downloads/IMG_1395.jpeg"
    result = check_flip(template_path, test_path)
    print(result)
