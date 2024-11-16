from ultralytics import YOLO
from generateXml import GenerateJpgAndXml
import numpy as np
import os
import cv2

# 加载yolov8模型
model = YOLO(r"C:\Users\lhb\Desktop\xingyu_test\small-object\best.pt")
# 修改为自己的标签名
label_dict = {0: 'holes', 1: 'joints', 2: 'press', 3: 'cut', 4: 'rubber', 5: 'screw', 6: 'oneu', 7: 'twou', 8: '40ngpress'}
parent_name = label_dict[0]

yolov8_xml = GenerateJpgAndXml(parent_name, label_dict)

# 指定图片所在文件夹的路径
image_folder_path = r'C:\Users\lhb\Desktop\train'

# 获取文件夹中所有的文件名
file_names = os.listdir(image_folder_path)

# 遍历每个文件
for file_name in file_names:
    # 判断是否是图片文件
    if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        # 图片的完整路径
        image_path = os.path.join(image_folder_path, file_name)

        # 使用OpenCV读取图片
        img = cv2.imread(image_path)

        # Perform object detection on an image using the model
        results = model.predict(source=img,
                                conf=0.1,
                                max_det=300,
                                iou=0.4,
                                half=True,
                                imgsz=640
                                )

        # print(results)
        for result in results:
            xyxy = result.to("cpu").numpy().boxes.xyxy
            print(result)
            # 假设 xyxy, conf 和 cls 分别是三个 NumPy 数组
            conf = result.to("cpu").numpy().boxes.conf
            cls = result.to("cpu").numpy().boxes.cls
            conf_expanded = np.expand_dims(conf, axis=1)  # 在轴 1 上扩充
            cls_expanded = np.expand_dims(cls, axis=1)  # 在轴 1 上扩充
            xyxy = xyxy.astype(np.int32)

            # 使用 numpy.concatenate() 在轴 1 上拼接数组
            concatenated_array = np.concatenate((cls_expanded, conf_expanded, xyxy), axis=1)
            print(concatenated_array)

            yolov8_xml.generatr_xml(img, concatenated_array)
            print(concatenated_array)

    print('-' * 50)
