import logging
import grpc
import detection_pb2, detection_pb2_grpc

def run_detection(path, ip):

    with grpc.insecure_channel(ip) as channel:
        stub = detection_pb2_grpc.DetectServiceStub(channel)
        request = detection_pb2.DetectRequest(imagePath=path)
        rep = stub.DetectFeaturesYolo(request)
        print(rep)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ip = "192.168.1.76:50052"
    paths = [
        "/Volumes/TU280Pro/ym/兴宇/xingyu_test/xingyu/20231130-084345.jpg"
    ]
    run_detection(paths, ip)