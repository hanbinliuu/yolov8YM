from typing import Dict

import cv2
import numpy as np

from supervision.draw.color import Color
from supervision.geometry.dataclasses import Point, Rect, Vector
from supervision.tools.detections import Detections

class LineCounter:
    def __init__(self, start: Point, end: Point):
        """
        Initialize a LineCounter object.

        :param start: Point : The starting point of the line.
        :param end: Point : The ending point of the line.
        """
        self.vector = Vector(start=start, end=end)
        self.tracker_state: Dict[str, bool] = {}
        self.in_count: int = 0
        self.out_count: int = 0

        # 自己加的
        self.in_count2: int = 0
        self.out_count2: int = 0

    # 自己加的
    def reset(self):
        # reset the tracker state and counts
        self.in_count: int = 0
        self.out_count: int = 0
        self.in_count2: int = 0
        self.out_count2: int = 0

    def update(self, detections: Detections):
        """
        Update the in_count and out_count for the detections that cross the line.

        :param detections: Detections : The detections for which to update the counts.
        """

        track = []
        for xyxy, confidence, class_id, tracker_id in detections:
            # handle detections with no tracker_id
            if tracker_id is None:
                continue

            # we check if all four anchors of bbox are on the same side of vector
            x1, y1, x2, y2 = xyxy
            anchors = [
                Point(x=x1, y=y1),
                Point(x=x1, y=y2),
                Point(x=x2, y=y1),
                Point(x=x2, y=y2),
            ]
            triggers = [self.vector.is_in(point=anchor) for anchor in anchors]

            # detection is partially in and partially out, 触线后，有一部分在线上，有一部分在线下
            if len(set(triggers)) == 2:
                continue

            tracker_state = triggers[0]
            # print('tracker_state: ', tracker_state)

            # handle new detection
            if tracker_id not in self.tracker_state:  # false
                self.tracker_state[tracker_id] = tracker_state
                continue

            # handle detection on the same side of the line
            if self.tracker_state.get(tracker_id) == tracker_state:
                continue

            # print(tracker_id)
            self.tracker_state[tracker_id] = tracker_state
            if tracker_state:
                self.in_count += 1
            else:
                self.out_count += 1
            track.append(tracker_id)

        return self.out_count, track

    # 自己加的
    def update2(self, detections: Detections):
        """
        Update the in_count and out_count for the detections that cross the line.

        :param detections: Detections : The detections for which to update the counts.
        """

        global touched
        track = []
        trigger_records = {}  # Create an empty dictionary to store triggers for each tracker_id

        for xyxy, confidence, class_id, tracker_id in detections:

            # handle detections with no tracker_id
            if tracker_id is None:
                continue

            # we check if all four anchors of bbox are on the same side of vector
            x1, y1, x2, y2 = xyxy
            anchors = [
                Point(x=x1, y=y1),
                Point(x=x1, y=y2),
                Point(x=x2, y=y1),
                Point(x=x2, y=y2),
            ]
            triggers = [self.vector.is_in(point=anchor) for anchor in anchors]

            trigger_records[tracker_id] = triggers
            # detection is partially in and partially out
            if len(set(triggers)) == 2:
                continue

            tracker_state = triggers[0]
            # handle new detection
            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = tracker_state
                continue

            # handle detection on the same side of the line
            if self.tracker_state.get(tracker_id) == tracker_state:
                continue

            self.tracker_state[tracker_id] = tracker_state
            if tracker_state:
                self.in_count2 += 1
            else:
                self.out_count2 += 1
            track.append(tracker_id)

        # 融合所有检测到的数据
        all_triggers = [triggers for triggers in trigger_records.values()]
        merged_triggers = [trigger for sublist in all_triggers for trigger in sublist]
        grouped_triggers = [merged_triggers[i:i + 4] for i in range(0, len(merged_triggers), 4)]
        result = any(len(set(group)) == 2 for group in grouped_triggers)

        if result:
            touched = True
        else:
            touched = False

        return self.out_count2, track, touched

class LineCounterAnnotator:
    def __init__(
            self,
            thickness: float = 2,
            color: Color = Color.white(),
            text_thickness: float = 2,
            text_color: Color = Color.black(),
            text_scale: float = 0.5,
            text_offset: float = 1.5,
            text_padding: int = 10,
    ):
        """
        Initialize the LineCounterAnnotator object with default values.

        :param thickness: float : The thickness of the line that will be drawn.
        :param color: Color : The color of the line that will be drawn.
        :param text_thickness: float : The thickness of the text that will be drawn.
        :param text_color: Color : The color of the text that will be drawn.
        :param text_scale: float : The scale of the text that will be drawn.
        :param text_offset: float : The offset of the text that will be drawn.
        :param text_padding: int : The padding of the text that will be drawn.
        """
        self.thickness: float = thickness
        self.color: Color = color
        self.text_thickness: float = text_thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_offset: float = text_offset
        self.text_padding: int = text_padding

    def annotate(self, frame: np.ndarray, line_counter: LineCounter) -> np.ndarray:
        """
        Draws the line on the frame using the line_counter provided.

        :param frame: np.ndarray : The image on which the line will be drawn
        :param line_counter: LineCounter : The line counter that will be used to draw the line
        :return: np.ndarray : The image with the line drawn on it
        """
        cv2.line(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            line_counter.vector.end.as_xy_int_tuple(),
            self.color.as_bgr(),
            self.thickness,
            lineType=cv2.LINE_AA,
            shift=0,
        )
        cv2.circle(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            radius=5,
            color=self.text_color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            frame,
            line_counter.vector.end.as_xy_int_tuple(),
            radius=5,
            color=self.text_color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        out_text = f"out: {line_counter.out_count}"

        (out_text_width, out_text_height), _ = cv2.getTextSize(
            out_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )

        out_text_x = int(
            (line_counter.vector.end.x + line_counter.vector.start.x - out_text_width)
            / 2
        )
        out_text_y = int(
            (line_counter.vector.end.y + line_counter.vector.start.y + out_text_height)
            / 2
            + self.text_offset * out_text_height
        )

        out_text_background_rect = Rect(
            x=out_text_x,
            y=out_text_y - out_text_height,
            width=out_text_width,
            height=out_text_height,
        ).pad(padding=self.text_padding)

        cv2.rectangle(
            frame,
            out_text_background_rect.top_left.as_xy_int_tuple(),
            out_text_background_rect.bottom_right.as_xy_int_tuple(),
            self.color.as_bgr(),
            -1,
        )

        cv2.putText(
            frame,
            out_text,
            (out_text_x, out_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )

        # 下面自己加的
        out_text2 = f"out: {line_counter.out_count2}"

        (out_text_width2, out_text_height2), _ = cv2.getTextSize(
            out_text2, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )


        out_text_x2 = int(
            (line_counter.vector.end.x + line_counter.vector.start.x - out_text_width2)
            / 2 + 60
        )
        out_text_y2 = int(
            (line_counter.vector.end.y + line_counter.vector.start.y + out_text_height2)
            / 2 + 60
            + self.text_offset * out_text_height2
        )

        out_text_background_rect2 = Rect(
            x=out_text_x2,
            y=out_text_y2 - out_text_height2,
            width=out_text_width2,
            height=out_text_height2,
        ).pad(padding=self.text_padding)

        cv2.rectangle(
            frame,
            out_text_background_rect2.top_left.as_xy_int_tuple(),
            out_text_background_rect2.bottom_right.as_xy_int_tuple(),
            self.color.as_bgr(),
            -1,
        )

        cv2.putText(
            frame,
            out_text2,
            (out_text_x2, out_text_y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )