from ultralytics import YOLO

def count_objects_in_images(image_paths, confidence_threshold=0.5):
    # 初始化模型
    model = YOLO("/model/yolov8cuthole2.pt")

    # 获取模型的类别名称列表
    names = model.names

    total_class_count = {}  # 创建一个总的目标数量统计字典

    for image_path in image_paths:
        # 使用加载的模型进行目标检测
        results = model.predict(image_path, save=False, imgsz=640, conf=confidence_threshold)

        # 遍历所有类别名称，并统计每个类别的目标数量
        for class_name in names:
            class_id = list(names).index(class_name)
            count = results[0].boxes.cls.tolist().count(class_id)

            # 如果该类别的数量大于0，则将数量累加到总的字典中
            if count > 0:
                if class_name in total_class_count:
                    total_class_count[class_name] += count
                else:
                    total_class_count[class_name] = count

    return total_class_count  # 返回合并后的总的目标数量统计字典


if __name__ == '__main__':
    image_paths = ["/Users/hanbinliu/PycharmProjects/datasets/cutpresshole3/images/train/IMG_2261.jpeg",
                   "/Users/hanbinliu/PycharmProjects/datasets/cutpresshole3/images/train/IMG_2262.jpeg"]

    total_class_count = count_objects_in_images(image_paths)
    print(total_class_count)
