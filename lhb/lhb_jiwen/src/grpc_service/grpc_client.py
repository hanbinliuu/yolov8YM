import logging
import cv2
import grpc

from lhb_jiwen.protos import detect_pb2, detect_pb2_grpc


def read_bytes_img(frame: str) -> bytes:
    ''' frame Into bytes '''
    # 图像编码
    success, img_encoded = cv2.imencode('.jpg', frame)
    # 图像转换为bytes
    img_encoded_bytes = img_encoded.tobytes()
    return img_encoded_bytes

def generate_frames():
    cap = cv2.VideoCapture("/Users/hanbinliu/Desktop/test4.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_byte = read_bytes_img(frame)
        yield detect_pb2.FrameRequest(origin_frame=frame_byte)


def run_detection(ip):

    with grpc.insecure_channel(ip) as channel:
        stub = detect_pb2_grpc.DetectServiceStub(channel)
        responses = stub.streamVideo(generate_frames())
        for response in responses:
            print("is_key_frame:", response.is_key_frame, "has_piece:", response.has_piece, "part_result_list:",
                  response.part_result_list)

def run_detection2(ip):

    """ 测试单向rpc """

    with grpc.insecure_channel(ip) as channel:
        stub = detect_pb2_grpc.DetectServiceStub(channel)
        request = detect_pb2.AlgorithmConfigRequest(model_file_path='test')
        response = stub.updateAlgorithmConfig(request)
        print("Response code:", response.code)
        print("Response message:", response.message)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ip = "192.168.6.94:50052"
    run_detection(ip)
