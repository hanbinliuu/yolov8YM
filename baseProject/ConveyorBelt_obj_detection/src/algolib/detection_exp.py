import copy
from typing import List
import numpy as np
from tqdm import tqdm

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink

from ConveyorBelt_obj_detection.src.algolib.flip_detection import check_flip
from ultralytics import YOLO


from ConveyorBelt_obj_detection.src.algolib.line_counter import LineCounter,LineCounterAnnotator
from ConveyorBelt_obj_detection.src.algolib.utilis import find_corresponding_tracker_ids, calculate_center_points, \
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

        self.source_vedio = "/Users/hanbinliu/Desktop/test3.mp4"
        self.video_info = VideoInfo.from_video_path(self.source_vedio)
        self.generator = get_video_frames_generator(self.source_vedio)

        self.output = "/Users/hanbinliu/Desktop/output2.mp4"

    @staticmethod
    def filter_detections_class_id(detections, classid):
        mask = np.array([class_id in classid for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

    @staticmethod
    def filter_detections_by_tracker_id(detections, tracker_ids):
        mask = np.array([tracker_id is not None for tracker_id in tracker_ids], dtype=bool)
        detections.filter(mask=mask, inplace=True)

    @staticmethod
    def track_update_id(detections, byte_tracker, frame):
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(id)
        return id

    # 标注label--，可视化信息
    def label(self, detections):
        return [
                f"#{tracker_id} {self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]

    def process_video(self, partnumber):

        global centerx_template, centery_template, selected_data
        hole_xyxy_dict = {}
        flip_result_dict = {}
        corresponding_tracker_ids_all = []

        # todo 平板件中心点坐标后面能不能固定下来, 检测一次模版耗时0.11s
        if self.model_path == "./mod/yolov8_hole.pt":
            test_result = self.model('./lhb_exp/jiwen/template_img/template.jpeg', classes=[1], device='mps')
            hole_detection = detection_result(test_result)
            # 找四个中心点
            centerx_template, centery_template = calculate_center_points(hole_detection.xyxy)

        with VideoSink(self.output, self.video_info) as sink:
            for frame in tqdm(self.generator, total=self.video_info.total_frames):

                results = self.model(frame, classes=[1, partnumber], device='mps')
                detections = detection_result(results)
                detections2 = copy.deepcopy(detections)

                self.filter_detections_class_id(detections=detections, classid=[1])
                self.filter_detections_class_id(detections=detections2, classid=[partnumber])

                tracker_nuts_id = self.track_update_id(detections=detections, byte_tracker=self.byte_tracker, frame=frame)
                tracker_part_id = self.track_update_id(detections=detections2, byte_tracker=self.byte_tracker_part, frame=frame)

                self.filter_detections_by_tracker_id(detections=detections, tracker_ids=tracker_nuts_id)
                self.filter_detections_by_tracker_id(detections=detections2, tracker_ids=tracker_part_id)

                labels = self.label(detections=detections)
                labels2 = self.label(detections=detections2)

                # 确认零件id和nuts的id对应关系 ---dictionary
                corresponding_tracker_ids = find_corresponding_tracker_ids(detections,detections2,detections2.tracker_id,detections.tracker_id)
                corresponding_tracker_ids_keys = list(corresponding_tracker_ids.keys())
                tracker_part_id_diff = list(set(tracker_part_id).difference(set(corresponding_tracker_ids_keys)))
                print(f'corresponding_tracker_ids,tracker_part_id_diff: {corresponding_tracker_ids}, {tracker_part_id_diff}')

                # 判断正反面
                full_hole_id = list(detections.tracker_id)  # 所有孔的id
                full_part_id = list(detections2.tracker_id)  # 所有零件的id

                # todo 可以设置必须超过某个位置，比如过了中心点开始检测
                if self.model_path == "./mod/yolov8_hole.pt" and len(full_part_id) > 0 and len(corresponding_tracker_ids) > 0:
                    for id in full_part_id:
                        select_hole_id = corresponding_tracker_ids.get(id)  # 找到零件对应的孔id
                        if select_hole_id is None:
                            continue
                        if len(select_hole_id) == 4 and id not in hole_xyxy_dict.keys():
                            # 根据孔id找对应的坐标位置
                            hole_xyxys = np.array([list(detections.xyxy[full_hole_id.index(id)]) for id in select_hole_id])
                            centerx_test, centery_test = calculate_center_points(hole_xyxys)
                            hole_xyxy_dict[id] = 'taken'

                            # 1 需要翻转 0 不需要翻转
                            try:
                                if check_flip(centerx_template, centery_template, centerx_test, centery_test):
                                    flip_result_dict[id] = 0
                                else:
                                    flip_result_dict[id] = 1
                                print(f'flip_result_dict:{flip_result_dict}')
                            except Exception as e:
                                print(f"Error occurred for sample {id}: {e}")
                                pass

                # 整个零件过线后才开始统计
                nuts_out, nuts_out_tracker_id = self.line_counter.update(detections=detections)
                frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=labels)
                part_out, part_out_tracker_id, part_touch_line = self.line_counter.update2(detections=detections2)
                frame2 = self.box_annotator.annotate(frame=frame, detections=detections2, labels=labels2)
                for img in [frame, frame2]:
                    self.line_annotator.annotate(frame=img, line_counter=self.line_counter)

                # 触线到出线，统计corresponding_tracker多帧，增加容错率
                corresponding_tracker_ids_all.append(corresponding_tracker_ids)

                # 没有螺母情况
                if len(part_out_tracker_id) == 1 and part_out_tracker_id[0] in tracker_part_id_diff:

                    # xyxy_idx = full_partid.index(part_out_tracker_id[0])
                    # partCoordinate = list(detections2.xyxy[xyxy_idx])
                    selected_data = {'part_code': partnumber, 'partId': part_out_tracker_id, 'nut_num': 0}

                    # 如果是平板件，需要加上翻转信息
                    if self.model_path == "./mod/yolov8_hole.pt" and part_out_tracker_id[0] in flip_result_dict:
                        flip = flip_result_dict[part_out_tracker_id[0]]
                        selected_data.update({'flip': flip})

                        # 过线pop缓存区
                        flip_result_dict.pop(part_out_tracker_id[0])
                        hole_xyxy_dict.pop(part_out_tracker_id[0])

                    self.line_counter.reset()
                    if part_out_tracker_id in tracker_part_id_diff:
                        tracker_part_id_diff.remove(part_out_tracker_id)

                    print(f'selected_data:{selected_data}')

                # 有螺母情况
                if len(part_out_tracker_id) == 1 and part_out_tracker_id[0] in corresponding_tracker_ids.keys():
                    # xyxy_idx = full_partid.index(part_out_tracker_id[0])
                    # partCoordinate = list(detections2.xyxy[xyxy_idx])
                    all_result = find_most_common_value_lists(corresponding_tracker_ids_all)
                    print(f'all_result:{all_result}')
                    out_num = len(all_result[part_out_tracker_id[0]][0])
                    selected_data = {'part_code': partnumber, 'part_id': part_out_tracker_id, 'nut_num': out_num}

                    # update 正反
                    if self.model_path == "./mod/yolov8_hole.pt" and part_out_tracker_id[0] in flip_result_dict:
                        flip = flip_result_dict[part_out_tracker_id[0]]
                        selected_data.update({'flip': flip})
                        # 过线pop缓存区
                        flip_result_dict.pop(part_out_tracker_id[0])
                        hole_xyxy_dict.pop(part_out_tracker_id[0])

                    all_result.pop(part_out_tracker_id[0])
                    self.line_counter.reset()
                    print(f'selected_data:{selected_data}')

                # todo 同时出线的情况
                if len(part_out_tracker_id) > 1:
                    selected_data_union = []    # 用于存储多个零件的数据
                    all_result = find_most_common_value_lists(corresponding_tracker_ids_all)
                    print(f'all_result:{all_result}')


                    for id in part_out_tracker_id:
                        if id in tracker_part_id_diff and len(tracker_part_id_diff) != 0:
                            selected_data = {'part_code': partnumber, 'part_id': id, 'nut_num': 0}
                            tracker_part_id_diff.pop(id)
                            all_result.pop(id)

                        if len(tracker_part_id_diff) == 0 and id in corresponding_tracker_ids.keys():
                            out_num = len(all_result[id][0])
                            selected_data = {'part_code': partnumber, 'part_id': id,'nut_num': out_num}
                            all_result.pop(id)

                        # update正反情况
                        if self.model_path == "./mod/yolov8_hole.pt" and id in flip_result_dict:
                            flip = flip_result_dict[id]
                            selected_data.update({'flip': flip})
                            # 过线pop缓存区
                            flip_result_dict.pop(id)
                            hole_xyxy_dict.pop(id)

                        selected_data_union.append(selected_data)
                    print(f'selected_data_union:{selected_data_union}')

                # 没有检测到，reset
                if not detections and not detections2:
                    self.line_counter.reset()

                sink.write_frame(frame)


if __name__ == "__main__":
    model_path_hole = "./mod/yolov8_hole.pt"
    model_path_1 = "./mod/yolov8_nuts.pt"

    obj = TrackObject(model_path=model_path_1)
    obj.process_video(partnumber=3)
