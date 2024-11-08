import os

import grpc
import argparse
import concurrent.futures
import signal
import threading
import contextlib

import detection_pb2_grpc
import frozen_dir
from iot_lib.iot_logger import LoggerConfigurator
from lhb_xingyu.src.grpc_service.grpc_server import DetectService


@contextlib.contextmanager
def run_detection_server(host, port, model_path):
    # create and start grpc servicer
    options = [
        ('grpc.max_send_message_length', 500 * 2048 * 2048),
        ('grpc.max_receive_message_length', 500 * 2048 * 2048)
    ]
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10), options=options)
    algor_servicer = DetectService(model_path)
    detection_pb2_grpc.add_DetectServiceServicer_to_server(algor_servicer, server)
    boundport = server.add_insecure_port(f"{host}:{port}")
    server.start()

    try:
        yield server, boundport
    finally:
        logger.info(f"stop grpc server at {host}:{boundport}")
        server.stop(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="192.168.6.95", help='grpc server port')
    parser.add_argument('--port', type=str, default='50052', help='port')
    parser.add_argument('--model_path', type=str, default="./lhb/model/yolov8cuthole2.pt", help='path to the yolo')

    args = parser.parse_args()
    print(args)

    # stop event: ctrl+c
    stop_event = threading.Event()

    def signal_handler(signum):
        signal.signal(signum, signal.SIG_IGN)
        logger.warning("receive signal to quit")
        stop_event.set()

    # register the signal with the signal handler first
    signal.signal(signal.SIGINT, signal_handler)

    with run_detection_server(args.host, args.port, args.model_path) as (server, port):
        logger.info(f"grpc Server is listening at port :{port}")
        stop_event.wait()


if __name__ == '__main__':
    # 获取脚本所在的目录
    current_dir = frozen_dir.app_path()

    # 构建相对路径
    log_file = os.path.join(current_dir, 'logs', 'xingyu_app.log')
    log_settings_file = os.path.join(current_dir, 'settings', 'log_settings.json')
    logger = LoggerConfigurator(fname=log_settings_file, handlerFileName=log_file).get_logger(__name__)
    try:
        main()
    except Exception as ex:
        logger.exception("exception raised")
