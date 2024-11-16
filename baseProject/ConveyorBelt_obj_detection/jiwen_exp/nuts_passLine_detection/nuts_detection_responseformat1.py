import copy
import math
import time
import uuid
import os
import cv2
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.tools.detections import Detections, BoxAnnotator
from typing import List
import numpy as np

from ConveyorBelt_obj_detection.jiwen_exp.algo.line_counter import LineCounter,LineCounterAnnotator
from ultralytics import YOLO
from tqdm import tqdm


def find_xyxy_index(original_dict, value_to_find):
    found_key = None
    for key, value_list in original_dict.items():
        if all(item in value_list for item in value_to_find):
            found_key = key
            break

    if found_key is not None:
        keys_list = list(original_dict.keys())
        index_of_key = keys_list.index(found_key)
        return index_of_key
    else:
        return None


def find_position_of_key(data_dict, key_to_find):
    keys_list = list(data_dict.keys())
    if key_to_find in keys_list:
        return keys_list.index(key_to_find)
    else:
        return -1


def find_index_of_element(original_list, a):
    try:
        index = original_list.index(a)
        return index
    except ValueError:
        return None

def save_frame_with_random_name(frame, output_dir):
    # 生成随机文件名
    random_filename = str(uuid.uuid4()) + ".png"
    output_path = os.path.join(output_dir, random_filename)

    # 保存图像
    cv2.imwrite(output_path, frame)

def find_corresponding_tracker_ids(detections, detections2, tracker_part_id, tracker_nuts_id):
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


def find_coordinates_for_ids(result_dict, detections):
    coordinates_list = []  # List to store pairs of coordinates associated with the given tracker IDs

    for key, tracker_id_list in result_dict.items():
        coordinates = []
        for tracker_id in tracker_id_list:
            if tracker_id in detections.tracker_id:
                index = np.where(detections.tracker_id == tracker_id)[0][0]
                bbox = detections.xyxy[index]
                coordinates.append(bbox)
        if coordinates:  # Check if coordinates were found for the current key
            coordinates_list.append(coordinates)

    return coordinates_list


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))

def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)

def match_detections_with_tracks(
        detections: Detections,
        tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

class TrackObject:

    def __init__(self) -> None:
        self.model = YOLO("./mod/yolov8_nuts.pt")
        self.CLASS_NAMES_DICT = self.model.model.names

        self.LINE_START = Point(1400, 1080-50)
        self.LINE_END = Point(1400, 50)

        self.byte_tracker = BYTETracker(BYTETrackerArgs())
        self.byte_tracker2 = BYTETracker(BYTETrackerArgs())

        self.line_counter = LineCounter(start=self.LINE_START, end=self.LINE_END)

        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1)
        self.box_annotator2 = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1)

        self.line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)
        self.line_annotator2 = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)
        self.part_out_id = []
        self.first_part_crossed_time = None
        self.time_window = 3

    @staticmethod
    def filter_detections_class_id(detections, classid):
        mask = np.array([class_id in classid for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

    @staticmethod
    def filter_detections_by_tracker_id(detections, tracker_ids):
        mask = np.array([tracker_id is not None for tracker_id in tracker_ids], dtype=bool)
        detections.filter(mask=mask, inplace=True)

    def label(self, detections):
        return [
                f"#{tracker_id} {self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]

    def process_video(self, partnumber):

        cap = cv2.VideoCapture(1)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        all_part_out = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, classes=[1, partnumber], device='mps')

            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )
            detections2 = copy.deepcopy(detections)

            # 没有检测到，reset
            if len(detections) == 0 and len(detections2) == 0:
                self.line_counter.reset()

            self.filter_detections_class_id(detections=detections, classid=[1])
            self.filter_detections_class_id(detections=detections2, classid=[partnumber])

            tracks = self.byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_nuts_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_nuts_id)

            #
            tracks2 = self.byte_tracker2.update(
                output_results=detections2boxes(detections=detections2),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_part_id = match_detections_with_tracks(detections=detections2, tracks=tracks2)
            detections2.tracker_id = np.array(tracker_part_id)
            print(f'tracker_part_id:{tracker_part_id}')
            self.filter_detections_by_tracker_id(detections=detections, tracker_ids=tracker_nuts_id)
            self.filter_detections_by_tracker_id(detections=detections2, tracker_ids=tracker_part_id)

            labels = self.label(detections=detections)
            labels2 = self.label(detections=detections2)
            # 整个零件过线后才开始统计----------------------
            nuts_out, nuts_out_tracker_id = self.line_counter.update(detections=detections)
            frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=labels)
            part_out, part_out_tracker_id = self.line_counter.update2(detections=detections2)
            frame2 = self.box_annotator.annotate(frame=frame, detections=detections2, labels=labels2)
            for img in [frame, frame2]:
                self.line_annotator.annotate(frame=img, line_counter=self.line_counter)

            # 确认零件和nuts的id对应关系---dictionary
            corresponding_tracker_ids = find_corresponding_tracker_ids(
                    detections=detections,
                    detections2=detections2,
                    tracker_part_id=detections2.tracker_id,
                    tracker_nuts_id=detections.tracker_id
                )
            corresponding_tracker_ids_keys = list(corresponding_tracker_ids.keys())
            tracker_part_id_diff = list(set(tracker_part_id).difference(set(corresponding_tracker_ids_keys)))
            print(f'corresponding_tracker_ids: {corresponding_tracker_ids}')
            print(f'tracker_part_id_diff: {tracker_part_id_diff}')
            if len(part_out_tracker_id) != 0 and part_out_tracker_id[0] in tracker_part_id_diff:
                xyxy_idx = find_position_of_key(corresponding_tracker_ids, part_out_tracker_id[0])
                out_num = 0
                partCoordinate = list(detections2.xyxy[xyxy_idx])
                selected_data = {'class_id': partnumber,
                                'partId': part_out_tracker_id,
                                'nutNum': out_num,
                                'x': partCoordinate[0],
                                'y': partCoordinate[1],
                                'w': partCoordinate[2],
                                'h': partCoordinate[3],
                                'pass_line_time': time.time()
                                }
                self.line_counter.reset()
                if part_out_tracker_id in tracker_part_id_diff:
                    tracker_part_id_diff.remove(part_out_tracker_id)
                print(f'selected_data:{selected_data}')

            if len(part_out_tracker_id) != 0 and part_out_tracker_id[0] in corresponding_tracker_ids.keys():
                xyxy_idx = find_position_of_key(corresponding_tracker_ids, part_out_tracker_id[0])
                out_num = len(corresponding_tracker_ids[part_out_tracker_id[0]])
                partCoordinate = list(detections2.xyxy[xyxy_idx])

                selected_data = {'class_id': partnumber,
                                    'partId': part_out_tracker_id,
                                    'nutNum': out_num,
                                    'x': partCoordinate[0],
                                    'y': partCoordinate[1],
                                    'w': partCoordinate[2],
                                    'h': partCoordinate[3],
                                    'pass_line_time': time.time()
                                    }
                self.line_counter.reset()
                print(f'selected_data:{selected_data}')

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    obj = TrackObject()
    obj.process_video(partnumber=4)
