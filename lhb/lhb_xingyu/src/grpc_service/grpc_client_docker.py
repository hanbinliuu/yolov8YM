import grpc
from protos import detection_pb2_grpc, detection_pb2


def run_detection(path):

    with grpc.insecure_channel('192.168.6.152:50052') as channel:
        stub = detection_pb2_grpc.DetectServiceStub(channel)
        request = detection_pb2.DetectRequest(imagePath=path)
        rep = stub.DetectFeaturesYolo(request)
        print(rep)


if __name__ == '__main__':

    # paths = ['/app/share/test/enhanced_1700108059_40.jpg',
    #          '/app/share/test/enhanced_1700108712_2.jpg',
    #             '/app/share/test/enhanced_1700110387_13.jpg',
    #          ]
    # paths = [
    #          '/app/share/Image_95.png',
    #             '/app/share/enhanced2_17.jpg',
    #             '/app/share/Image_157.png',
    #             '/app/share/Image_25197.png',
    #             '/app/share/00.jpg',
    #             '/app/share/Image_9869.png',
    #             '/app/share/51.jpg',
    #
    #          ]

    paths = [
               '/app/share/xingyu/img_v3_0269_3f1eaa8a-9aed-41ce-8157-510f5835a65g.jpg'
             ]

    run_detection(paths)

#  开服务
# docker run -p 50052:50052 xingyu

# 挂载卷开服务
# docker run -v C:/Users/Administrator/Desktop/test:/app/share -p 50052:50052 xingyu

# 执行进入container内部
# docker exec -it d1cc85858a03 /bin/bash

# copy file into container（需要在外部terminal中执行）
# docker cp C:/Users/Administrator/Desktop/share 6366b287f1ee:/app




