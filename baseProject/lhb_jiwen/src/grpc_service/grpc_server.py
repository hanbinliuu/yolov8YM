import logging
import grpc
from concurrent import futures
import time

from lhb_jiwen.src.core.detectionController import TrackObject
from lhb_jiwen.src.iot_lib.stopwatch import StopWatch
from lhb_jiwen.protos import detect_pb2, detect_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class DetectServer(detect_pb2_grpc.DetectServiceServicer):

    def __init__(self):
        self._logger = logging.getLogger('detection grpc-server')
        self._logger.setLevel(logging.INFO)
        # 检测
        self.detecion = TrackObject(model_path="../../../model/yolov8n.pt")

    def streamVideo(self, request_iterator, context):
        self._logger.info('gRPC@detection receive request')
        log_cat = f"({self.streamVideo.__name__})"

        try:
            if not request_iterator:
                self._logger.warning(f"{log_cat} request is None")

            for frame_request in request_iterator:
                stopwatch = StopWatch()
                stopwatch.start()
                result, process_frame, is_key_frame, anchor_touched = self.detecion.process_video(
                    frame_request.origin_frame)

                stopwatch.stop()
                self._logger.info(f"{log_cat} using {stopwatch.elapsed_time}s")
                print("Is Key Frame:", is_key_frame, "Has Piece:", anchor_touched, "Part Result List:", result)
                yield detect_pb2.FrameResponse(processed_frame=process_frame,
                                               is_key_frame=is_key_frame,
                                               has_piece=anchor_touched,
                                               part_result_list=result)
        except Exception as e:
            self._logger.error(f"{log_cat} Exception: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            yield detect_pb2.FrameResponse()

    # 测试单向流
    def updateAlgorithmConfig(self, request: detect_pb2.AlgorithmConfigRequest, context):

        print('connect to server')
        return detect_pb2.CommonResponse(code=1, message='connect test success')


def run():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    detect_pb2_grpc.add_DetectServiceServicer_to_server(DetectServer(), server)
    print("start service...")
    server.add_insecure_port('[::]:50052')
    server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        print("stop service...")


if __name__ == '__main__':
    run()
