import os
import uuid

import cv2
import numpy as np
from collections import defaultdict


def save_frame_with_random_name(frame, output_dir):
    # 生成随机文件名
    random_filename = str(uuid.uuid4()) + ".png"
    output_path = os.path.join(output_dir, random_filename)

    # 保存图像
    cv2.imwrite(output_path, frame)

def find_corresponding_tracker_ids2(detections, detections2, tracker_part_id, tracker_nuts_id):
    corresponding_tracker_ids = {}  # Dictionary to store corresponding tracker_part_id and tracker_nuts_id pairs

    # Loop through each detection2 (part) in detections2
    for j, detection2 in enumerate(detections2.xyxy):
        # Create a list to store corresponding nuts_id for the current part
        corresponding_nuts_ids = []

        # Loop through each detection (nut) in detections
        for i, detection in enumerate(detections.xyxy):
            # Compute the coordinates of the top-left and bottom-right corners of the current detection (nut)
            x_min = detection[0]
            y_min = detection[1]
            x_max = detection[2]
            y_max = detection[3]

            # Compute the coordinates of the top-left and bottom-right corners of the current detection2 (part)
            x_min_part = detection2[0]
            y_min_part = detection2[1]
            x_max_part = detection2[2]
            y_max_part = detection2[3]

            # Check if detection from detections is inside detection2 bounding box
            if x_min_part <= x_min <= x_max_part and y_min_part <= y_min <= y_max_part \
                    and x_min_part <= x_max <= x_max_part and y_min_part <= y_max <= y_max_part:
                if i < len(tracker_nuts_id):  # Check if i is within the valid range of tracker_nuts_id
                    # Append the corresponding nuts_id to the list
                    corresponding_nuts_ids.append(tracker_nuts_id[i])

        # Check if any corresponding nuts_id found for the current part
        if corresponding_nuts_ids:
            corresponding_tracker_ids[tracker_part_id[j]] = corresponding_nuts_ids

    return corresponding_tracker_ids

# 优化上面代码
def find_corresponding_tracker_ids(detections, detections2, tracker_part_id, tracker_nuts_id):
    corresponding_tracker_ids = {}

    for j, detection2 in enumerate(detections2.xyxy):
        x_min_part, y_min_part, x_max_part, y_max_part = detection2

        # Calculate IoU between detection2 and all detections
        x_min = detections.xyxy[:, 0]
        y_min = detections.xyxy[:, 1]
        x_max = detections.xyxy[:, 2]
        y_max = detections.xyxy[:, 3]

        ious = (
            (np.minimum(x_max, x_max_part) - np.maximum(x_min, x_min_part)) *
            (np.minimum(y_max, y_max_part) - np.maximum(y_min, y_min_part))
        ) / (
            (x_max - x_min) * (y_max - y_min) +
            (x_max_part - x_min_part) * (y_max_part - y_min_part) -
            (np.minimum(x_max, x_max_part) - np.maximum(x_min, x_min_part)) *
            (np.minimum(y_max, y_max_part) - np.maximum(y_min, y_min_part))
        )

        # Find detections with IoU > 0 (overlap)
        valid_indices = np.where(ious > 0)[0]

        # Get corresponding nuts_ids for valid detections
        corresponding_nuts_ids = [tracker_nuts_id[i] for i in valid_indices if i < len(tracker_nuts_id)]

        if corresponding_nuts_ids:
            corresponding_tracker_ids[tracker_part_id[j]] = corresponding_nuts_ids

    return corresponding_tracker_ids

def find_position_of_key(data_dict, key_to_find):
    keys_list = list(data_dict.keys())
    if key_to_find in keys_list:
        return keys_list.index(key_to_find)
    else:
        return -1

def calculate_center_points(xyxys):
    center_points = np.zeros((xyxys.shape[0], 2), dtype=np.float32)
    center_points[:, 0] = (xyxys[:, 0] + xyxys[:, 2]) / 2  # x坐标
    center_points[:, 1] = (xyxys[:, 1] + xyxys[:, 3]) / 2  # y坐标
    centerx = center_points[:, 0].tolist()
    centery = center_points[:, 1].tolist()
    return centerx, centery

def find_most_common_value_lists(data):
    key_value_counts = defaultdict(lambda: defaultdict(int))

    for item in data:
        for key, value_list in item.items():
            sorted_value_list = sorted(value_list)
            value_tuple = tuple(sorted_value_list)
            key_value_counts[key][value_tuple] += 1

    most_common_lists = {}

    for key, value_counts in key_value_counts.items():
        most_common_value_list, most_common_count = max(value_counts.items(), key=lambda x: x[1])
        most_common_lists[key] = (most_common_value_list, most_common_count)

    return most_common_lists