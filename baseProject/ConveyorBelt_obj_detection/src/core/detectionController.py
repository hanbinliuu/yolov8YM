import copy
import logging
import cv2
from typing import List
import numpy as np
from tqdm import tqdm

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.tools.detections import Detections, BoxAnnotator
from ultralytics import YOLO

from ConveyorBelt_obj_detection.src.algolib.flip_detection import check_flip
from ConveyorBelt_obj_detection.src.algolib.line_counter import LineCounter, LineCounterAnnotator
from ConveyorBelt_obj_detection.src.algolib.utilis import find_corresponding_tracker_ids2, calculate_center_points, \
    find_most_common_value_lists


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

def detection_result(model_result):
    detections = Detections(
    xyxy=model_result[0].boxes.xyxy.cpu().numpy(),
    confidence=model_result[0].boxes.conf.cpu().numpy(),
    class_id=model_result[0].boxes.cls.cpu().numpy().astype(int)
    )
    return detections

class TrackObject:

    def __init__(self, model_path) -> None:
        self._logger = logging.getLogger('detection controller')

        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.CLASS_NAMES_DICT = self.model.model.names

        self.LINE_START = Point(50, 1200)
        self.LINE_END = Point(1200 - 50, 1200)

        # 有n类必须要有n个tracker
        self.byte_tracker = BYTETracker(BYTETrackerArgs())
        self.byte_tracker_part = BYTETracker(BYTETrackerArgs())

        self.line_counter = LineCounter(start=self.LINE_START, end=self.LINE_END)
        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1)
        self.line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)

        self.hole_xyxy_dict = {}
        self.flip_result_dict = {}
        self.corresponding_tracker_ids_all = []

        if self.model_path.endswith("hole.pt"):
            self.centerx_template, self.centery_template = self.process_template()

    def process_template(self):
        test_result = self.model('./lhb_exp/jiwen/template_img/template.jpeg', classes=[1], device='mps')
        hole_detection = detection_result(test_result)
        # 找四个中心点
        centerx_template, centery_template = calculate_center_points(hole_detection.xyxy)
        return centerx_template, centery_template

    @staticmethod
    def filter_detections_class_id(detections, classid):
        mask = np.array([class_id in classid for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

    @staticmethod
    def filter_detections_by_tracker_id(detections, tracker_ids):
        mask = np.array([tracker_id is not None for tracker_id in tracker_ids], dtype=bool)
        detections.filter(mask=mask, inplace=True)

    @staticmethod
    def track_update_id(detections, byte_tracker, frame_bytes):
        frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype="uint8"), cv2.IMREAD_COLOR)
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(id)
        return id

    # 标注label，可视化信息
    def label(self, detections):
        return [
                f"#{tracker_id} {self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]

    def process_video(self, frame_bytes):
        print("process_video:", frame_bytes)

        frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype="uint8"), cv2.IMREAD_COLOR)
        global selected_data
        selected_data_all = []

        # todo class需要修改
        import time
        start = time.time()
        results = self.model(frame, classes=[1, 3], device='mps')
        detections = detection_result(results)
        detections2 = copy.deepcopy(detections)

        self.filter_detections_class_id(detections=detections, classid=[1])
        self.filter_detections_class_id(detections=detections2, classid=[3])

        test_frame =read_bytes_img(frame)
        tracker_nuts_id = self.track_update_id(detections=detections, byte_tracker=self.byte_tracker, frame_bytes=test_frame)
        tracker_part_id = self.track_update_id(detections=detections2, byte_tracker=self.byte_tracker_part, frame_bytes=test_frame)

        self.filter_detections_by_tracker_id(detections, tracker_ids=tracker_nuts_id)
        self.filter_detections_by_tracker_id(detections2, tracker_ids=tracker_part_id)

        labels, labels2 = self.label(detections=detections), self.label(detections=detections2)

        # 确认零件id和nuts的id对应关系 ---dictionary
        corresponding_tracker_ids = find_corresponding_tracker_ids2(detections, detections2, detections2.tracker_id, detections.tracker_id)
        corresponding_tracker_ids_keys = list(corresponding_tracker_ids.keys())
        tracker_part_id_diff = list(set(tracker_part_id).difference(set(corresponding_tracker_ids_keys)))
        print(f'corresponding_tracker_ids,tracker_part_id_diff: {corresponding_tracker_ids}, {tracker_part_id_diff}')

        # 判断正反面
        full_hole_id = list(detections.tracker_id)  # 所有孔的id
        full_part_id = list(detections2.tracker_id)  # 所有零件的id

        # todo 可以设置必须超过某个位置，比如过了中心点开始检测
        if self.model_path.endswith("hole.pt") and \
                len(corresponding_tracker_ids) > 0:
            for id in full_part_id:
                select_hole_id = corresponding_tracker_ids.get(id)  # 找到零件对应的孔id
                if select_hole_id is None:
                    continue
                if len(select_hole_id) == 4 and id not in self.hole_xyxy_dict.keys():
                    # 根据孔id找对应的坐标位置
                    hole_xyxys = np.array([list(detections.xyxy[full_hole_id.index(id)]) for id in select_hole_id])
                    centerx_test, centery_test = calculate_center_points(hole_xyxys)
                    self.hole_xyxy_dict[id] = 'taken'

                    # 1 需要翻转 0 不需要翻转
                    try:
                        if check_flip(self.centerx_template, self.centery_template, centerx_test, centery_test):
                            self.flip_result_dict[id] = 0
                        else:
                            self.flip_result_dict[id] = 1
                        print(f'flip_result_dict:{self.flip_result_dict}')
                    except Exception as e:
                        print(f"Error occurred for sample {id}: {e}")
                        pass

        # 整个零件过线后才开始统计
        nuts_out, nuts_out_tracker_id = self.line_counter.update(detections=detections)
        frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        part_out, part_out_tracker_id, anchor_touched = self.line_counter.update2(detections=detections2)

        frame2 = self.box_annotator.annotate(frame=frame, detections=detections2, labels=labels2)
        for img in [frame, frame2]:
            self.line_annotator.annotate(frame=img, line_counter=self.line_counter)
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)

        # 统计corresponding_tracker多帧，增加容错率
        self.corresponding_tracker_ids_all.append(corresponding_tracker_ids)

        # 没有螺母情况
        if len(part_out_tracker_id) == 1 and part_out_tracker_id[0] in tracker_part_id_diff:
            # log 检测到有零件过线
            self._logger.info(f'part is passing the line!')
            selected_data = {'part_code': str(0), 'part_id': str(part_out_tracker_id), 'nut_num': int(0)}

            # 如果是平板件，需要加上翻转信息
            if self.model_path.endswith("hole.pt") and\
                    part_out_tracker_id[0] in self.flip_result_dict:
                flip = self.flip_result_dict[part_out_tracker_id[0]]
                print(f'flip:{flip}')
                selected_data.update({'face_direction': int(flip)})

                # 过线pop缓存区
                self.flip_result_dict.pop(part_out_tracker_id[0])
                self.hole_xyxy_dict.pop(part_out_tracker_id[0])

            selected_data_all.append(selected_data)
            self.line_counter.reset()
            if part_out_tracker_id in tracker_part_id_diff:
                tracker_part_id_diff.remove(part_out_tracker_id)

            processed_frame = read_bytes_img(frame)
            is_key_frame = True

            return selected_data_all, processed_frame, is_key_frame,anchor_touched

        # 有螺母情况
        if len(part_out_tracker_id) == 1 and part_out_tracker_id[0] in corresponding_tracker_ids.keys():
            self._logger.info(f'part is passing the line!')

            all_result = find_most_common_value_lists(self.corresponding_tracker_ids_all)
            print(f'all_result:{all_result}')
            out_num = len(all_result[part_out_tracker_id[0]][0])
            selected_data = {'part_code': str(0), 'part_id': str(part_out_tracker_id), 'nut_num': int(out_num)}

            # update 正反
            if self.model_path.endswith("hole.pt") and part_out_tracker_id[0] in self.flip_result_dict:
                flip = self.flip_result_dict[part_out_tracker_id[0]]
                selected_data.update({'face_direction': int(flip)})

                # 过线pop缓存区
                self.flip_result_dict.pop(part_out_tracker_id[0])
                self.hole_xyxy_dict.pop(part_out_tracker_id[0])

            selected_data_all.append(selected_data)
            all_result.pop(part_out_tracker_id[0])
            self.line_counter.reset()

            processed_frame = read_bytes_img(frame)
            is_key_frame = True

            return selected_data_all, processed_frame, is_key_frame, anchor_touched

        # todo 同时出线的情况
        if len(part_out_tracker_id) > 1:

            self._logger.info(f'part is passing the line!')
            selected_data_union = []  # 用于存储多个零件的数据
            all_result = find_most_common_value_lists(self.corresponding_tracker_ids_all)
            print(f'all_result:{all_result}')

            for id in part_out_tracker_id:
                if id in tracker_part_id_diff and len(tracker_part_id_diff) != 0:
                    selected_data = {'part_code': str(0), 'part_id': str(id), 'nut_num': int(0)}
                    tracker_part_id_diff.pop(id)
                    all_result.pop(id)

                if len(tracker_part_id_diff) == 0 and id in corresponding_tracker_ids.keys():
                    out_num = len(all_result[id][0])
                    selected_data = {'part_code': str(0), 'part_id': str(id), 'nut_num': int(out_num)}
                    all_result.pop(id)

                # update正反情况
                if self.model_path.endswith("hole.pt") and id in self.flip_result_dict:
                    flip = self.flip_result_dict[id]
                    selected_data.update({'face_direction': int(flip)})
                    # 过线pop缓存区
                    self.flip_result_dict.pop(id)
                    self.hole_xyxy_dict.pop(id)
                selected_data_union.append(selected_data)

            processed_frame = read_bytes_img(frame)
            is_key_frame = True
            return selected_data_union, processed_frame, is_key_frame,anchor_touched

        # 没有检测到，reset
        if not detections and not detections2:
            self.line_counter.reset()

        # 其他情况不是关键帧
        is_key_frame = False
        return selected_data_all , read_bytes_img(frame), is_key_frame, anchor_touched

def read_bytes_img(frame: str) -> bytes:
    ''' frame Into bytes '''
    # 图像编码
    # print(f'frame:{frame}')
    success, img_encoded = cv2.imencode('.jpg', frame)
    # 图像转换为bytes
    img_encoded_bytes = img_encoded.tobytes()
    return img_encoded_bytes


if __name__ == "__main__":

    # 视频转换成bytes
    video_path = "/Users/hanbinliu/Desktop/test3.mp4"
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAtouchME_COUNT))

    obj = TrackObject(model_path="./mod/yolov8_nuts.pt")
    with tqdm(total=total_frames, desc="Processing Frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            img_encoded_bytes = read_bytes_img(frame)
            result, process_frame, is_key_frame, anchor_touched = obj.process_video(frame_bytes=img_encoded_bytes)
            print('result:', result, 'is_key_frame:', is_key_frame,'anchor_touched:', anchor_touched)
            pbar.update(1)