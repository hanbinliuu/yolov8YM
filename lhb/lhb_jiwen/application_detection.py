import grpc
import argparse
import concurrent.futures
import signal
import threading
import contextlib

from lhb_jiwen.protos import detect_pb2_grpc
from lhb_jiwen.src.grpc_service.grpc_server import DetectServer
from lhb_jiwen.src.iot_lib.iot_logger import LoggerConfigurator


@contextlib.contextmanager
def run_detection_server(host, port):
    # create and start grpc servicer
    options = [
        ('grpc.max_send_message_length', 500 * 2048 * 2048),
        ('grpc.max_receive_message_length', 500 * 2048 * 2048)
    ]
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10), options=options)
    algor_servicer = DetectServer()
    detect_pb2_grpc.add_DetectServiceServicer_to_server(algor_servicer, server)
    boundport = server.add_insecure_port(f"{host}:{port}")
    server.start()

    try:
        yield server, boundport
    finally:
        logger.info(f"stop grpc server at {host}:{boundport}")
        server.stop(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="192.168.2.41", help='grpc server port')
    parser.add_argument('--port', type=str, default='50052', help='log level')
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

    with run_detection_server(args.host, args.port) as (server, port):
        logger.info(f"grpc Server is listening at port :{port}")
        stop_event.wait()


if __name__ == '__main__':
    log_file = "logs/jiwen_app.log"
    logger = LoggerConfigurator(fname="settings/log_settings.json", handlerFileName=log_file).get_logger(
        __name__)
    try:
        main()
    except Exception as ex:
        logger.exception("exception raised")
