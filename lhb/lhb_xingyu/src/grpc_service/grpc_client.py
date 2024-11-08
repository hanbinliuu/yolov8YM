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
    ip = "0.0.0.0:50052"
    paths = [
        "/Users/hanbinliu/PycharmProjects/datasets/cutpresshole3/images/train/IMG_2261.jpeg",
        "/Users/hanbinliu/PycharmProjects/datasets/cutpresshole3/images/train/IMG_2222.jpeg",
        "/Users/hanbinliu/PycharmProjects/datasets/cutpresshole3/images/train/IMG_2263.jpeg"
    ]
    run_detection(paths, ip)