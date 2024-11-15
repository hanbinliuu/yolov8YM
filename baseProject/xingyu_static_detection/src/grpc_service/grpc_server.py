import logging
import time
import grpc
from concurrent import futures

from xingyu_static_detection.src.iot_lib.stopwatch import StopWatch
from protos import detection_pb2, detection_pb2_grpc
from xingyu_static_detection.src.algolib.detection import count_objects_in_images
from ultralytics import YOLO

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class DetectService(detection_pb2_grpc.DetectServiceServicer):
    def __init__(self, modpath):
        self._logger = logging.getLogger('detection grpc-server')
        # 模型位置写死
        self.yolo = YOLO(modpath)

    def DetectFeaturesYolo(self, request:detection_pb2.DetectRequest, context):

        self._logger.info('gRPC@detection receive request')
        log_cat = f"({self.DetectFeaturesYolo.__name__})"

        if not request.imagePath:
            msg = 'Invalid image data: No image paths provided.'
            self._logger.warning(f"{log_cat} {msg}")
            return detection_pb2.DetectResponse(error=msg)  # 返回错误响应

        # 指定request
        image_paths = request.imagePath
        stopwatch = StopWatch()
        stopwatch.start()
        detection_result = count_objects_in_images(self.yolo, image_paths)
        stopwatch.stop()
        self._logger.info(f"{log_cat} using {stopwatch.elapsed_time}s")
        self._logger.info(f"{log_cat} result: {detection_result}")

        return detection_pb2.DetectResponse(feature_counts=detection_result)


def server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    detection_pb2_grpc.add_DetectServiceServicer_to_server(DetectService(), server)
    server.add_insecure_port('[::]:50052')
    print("start service...")
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        print("stop service...")


if __name__ == '__main__':
    server()
