import copy
import numpy as np
import math
import matplotlib.pyplot as plt

# 四个孔找找三个（到时候可以省略，直接标注三个孔）
def distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

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

def find_shortest_edge(points):
    # 计算三角形三条边的长度

    if len(points) < 3:
        pass
    else:
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

# 三角形旋转矩阵
def rotataion_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])

# 平移三角形到y=0
def translate_triangle(triangle):
    # Calculate edge lengths
    edge_lengths = [distance(triangle[i], triangle[(i + 1) % 3]) for i in range(3)]

    # Find the index of the shortest edge
    shortest_edge_index = np.argmin(edge_lengths)

    # Determine the translation amount
    translation_amount = -triangle[shortest_edge_index][1]

    # Translate the triangle to align the shortest edge with y=0
    translated_triangle = np.copy(triangle)
    translated_triangle[:, 1] += translation_amount
    return translated_triangle


# def translate_triangle2(triangle):
#     # Find the shortest edge and its endpoints
#     shortest_edge = float('inf')
#     shortest_edge_index = None
#
#     for i in range(3):
#         edge_length = distance(triangle[i], triangle[(i + 1) % 3])
#
#         if edge_length < shortest_edge:
#             shortest_edge = edge_length
#             shortest_edge_index = i
#
#     # Determine the translation amount
#     translation_amount = -triangle[shortest_edge_index][1]
#
#     # Translate the triangle
#     translated_triangle = np.array([(x, y + translation_amount) for x, y in triangle], dtype=np.float32)
#     return translated_triangle


# 旋转三角形到正确位置
def rotate_to_horizontal(points, plot=True):
    # 找到最短边的两个顶点
    shortest_vertex, other_vertex = find_shortest_edge(points)

    # 计算最短边的角度
    angle = np.arctan2(other_vertex[1] - shortest_vertex[1], other_vertex[0] - shortest_vertex[0])

    # 计算旋转矩阵
    rotation_matrix = rotataion_matrix(angle)

    # 进行旋转(原始点位旋转)
    rotated_points = np.dot(points, rotation_matrix)
    rotated_transform_points = translate_triangle(rotated_points)

    if np.any(rotated_transform_points[:, 1] < 0):
        rotated_transform_points_new = np.dot(rotated_transform_points, rotataion_matrix(np.pi))
        plot_triangle(rotated_transform_points_new, 'Rotated to Horizontal')
        return rotated_transform_points_new
    else:
        plot_triangle(rotated_transform_points, 'Rotated to Horizontal')
        return rotated_transform_points



def plot_triangle(points, title):
    plt.figure()
    plt.plot(points[:, 0], points[:, 1], 'bo-')
    plt.plot([points[0, 0], points[-1, 0]], [points[0, 1], points[-1, 1]], 'bo-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()

# def triangle_selection(centerx, centery):
#     zip_data = np.float32([[x, y] for x, y in zip(centerx, centery)])
#     distances = find_closest_points(zip_data)
#     removal_idx = list(find_idx(distances))[0]
#
#     # todo pop 出问题了
#     centerx.pop(removal_idx)
#     centery.pop(removal_idx)
#     coordinates = np.float32([[x, y] for x, y in zip(centerx, centery)])
#     return coordinates

# todo 找洞会有问题，到时候标注螺母后可以解决
def triangle_selection(centerx, centery):
    centerx_copy = centerx.copy()
    centery_copy = centery.copy()

    zip_data = np.float32([[x, y] for x, y in zip(centerx_copy, centery_copy)])
    distances = find_closest_points(zip_data)
    removal_idx = list(find_idx(distances))[0]

    centerx_copy.pop(removal_idx)
    centery_copy.pop(removal_idx)

    coordinates = np.float32([[x, y] for x, y in zip(centerx_copy, centery_copy)])
    return coordinates


def distance2(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 计算三角形的角度
def calculate_angle(x1, y1, x2, y2, x3, y3):
    a = distance2(x2, y2, x3, y3)
    b = distance2(x1, y1, x3, y3)
    c = distance2(x1, y1, x2, y2)
    return math.degrees(math.acos((a**2 + b**2 - c**2) / (2 * a * b)))

# 最短边的中心点x坐标和最大角x坐标
def calculate_triangle_info(triangle):
    edge_lengths = [
        distance(triangle[0], triangle[1]),
        distance(triangle[1], triangle[2]),
        distance(triangle[2], triangle[0]),
    ]

    min_edge_index = edge_lengths.index(min(edge_lengths))
    vertex_indices = [min_edge_index, (min_edge_index + 1) % 3]
    shortest_edge_vertices = [triangle[i] for i in vertex_indices]
    min_edge_x = (shortest_edge_vertices[0][0] + shortest_edge_vertices[1][0]) / 2

    angles = [
        calculate_angle(*triangle[0], *triangle[1], *triangle[2]),
        calculate_angle(*triangle[1], *triangle[2], *triangle[0]),
        calculate_angle(*triangle[2], *triangle[0], *triangle[1])
    ]
    max_angle_index = angles.index(max(angles))
    max_angle_xcoord = triangle[max_angle_index - 1][0]

    return min_edge_x, max_angle_xcoord

# 判断是否同大或同小
def are_max_angle_x_coordinates_smaller_than_half_horizontal_edge(triangle1, triangle2):
    min_edge_x1, max_angle_xcoord1 = calculate_triangle_info(triangle1)
    min_edge_x2, max_angle_xcoord2 = calculate_triangle_info(triangle2)

    result = (max_angle_xcoord1 < min_edge_x1 and max_angle_xcoord2 < min_edge_x2) \
             or (max_angle_xcoord1 > min_edge_x1 and max_angle_xcoord2 > min_edge_x2)

    return result

# check flip
def check_flip(centerx_template, centery_template, centerx_sample, centery_sample):
    """ 判断是不是翻转 """

    template_cord = triangle_selection(centerx_template, centery_template)
    test_cord = triangle_selection(centerx_sample, centery_sample)

    rotated_points = rotate_to_horizontal(template_cord)
    rotated_points2 = rotate_to_horizontal(test_cord)

    if are_max_angle_x_coordinates_smaller_than_half_horizontal_edge(rotated_points, rotated_points2):
        return True
    else:
        return False
