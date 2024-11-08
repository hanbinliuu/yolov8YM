import copy

import cv2
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from typing import List
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

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

class TrackObject():

    def __init__(self) -> None:
        self.model = YOLO("./mod/yolov8_nuts.pt")
        self.CLASS_NAMES_DICT = self.model.model.names
        self.CLASS_ID = [1]
        self.CLASS_ID2 = [0]


        self.LINE_START = Point(1400, 1080-50)
        self.LINE_END = Point(1400, 50)

        self.byte_tracker = BYTETracker(BYTETrackerArgs())
        self.byte_tracker2 = BYTETracker(BYTETrackerArgs())

        self.line_counter = LineCounter(start=self.LINE_START, end=self.LINE_END)

        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1)
        self.box_annotator2 = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1)

        self.line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)
        self.line_annotator2 = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)


        self.num_consecutive_frames_without_detections = 0


    def process_video(self):
        cap = cv2.VideoCapture(1)  # Use the default camera (camera index 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(f"height: {height}, width: {width}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, classes=[0,1], device='mps')

            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )

            #
            detections2 = copy.deepcopy(detections)


            if len(detections) == 0:
                self.num_consecutive_frames_without_detections += 1
            else:
                self.num_consecutive_frames_without_detections = 0

            if self.num_consecutive_frames_without_detections >= 5:
                self.line_counter.reset()

            mask = np.array([class_id in self.CLASS_ID for class_id in detections.class_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)

            #
            mask2 = np.array([class_id in self.CLASS_ID2 for class_id in detections2.class_id], dtype=bool)
            detections2.filter(mask=mask2, inplace=True)

            tracks = self.byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )

            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)

            #
            tracks2 = self.byte_tracker2.update(
                output_results=detections2boxes(detections=detections2),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_id2 = match_detections_with_tracks(detections=detections2, tracks=tracks2)
            detections2.tracker_id = np.array(tracker_id2)


            mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)

            #
            mask2 = np.array([tracker_id2 is not None for tracker_id2 in detections2.tracker_id], dtype=bool)
            detections2.filter(mask=mask2, inplace=True)

            labels = [
                f"#{tracker_id} {self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]

            labels2 = [
                f"#{tracker_id} {self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections2
            ]

            self.line_counter.update(detections=detections)
            self.line_counter.update2(detections=detections2)

            frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=labels)
            frame2 = self.box_annotator.annotate(frame=frame, detections=detections2, labels=labels2)

            self.line_annotator.annotate(frame=frame, line_counter=self.line_counter)
            self.line_annotator.annotate(frame=frame2, line_counter=self.line_counter)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    obj = TrackObject()
    obj.process_video()
