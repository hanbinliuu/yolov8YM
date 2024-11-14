from ultralytics import YOLO


def count_objects_in_images(yolo_model, image_paths, confidence_threshold=0.7):
    paths = []

    for item in image_paths:
        path = item.strip('"')
        paths.append(path)

    # 获取模型的类别名称列表
    names = yolo_model.names
    print("names: ", names)

    total_class_count = {}  # 创建一个总的目标数量统计字典

    for image_path in paths:
        # 使用加载的模型进行目标检测
        results = yolo_model.predict(image_path, save=False, imgsz=640, conf=confidence_threshold)

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
    sorted_class_count = dict(sorted(total_class_count.items()))

    for key, value in sorted_class_count.items():
        if key in [2, 6] and value > 1:
            sorted_class_count[key] = 1

    return sorted_class_count


if __name__ == '__main__':
    yolo_model = YOLO(r"C:\Users\lhb\Desktop\yolov8_xingyu.pt")
    image_paths = [r"C:\Users\lhb\Desktop\img_v3_0269_a46bbbbf-10bb-4420-b38a-d91dd77de33g.jpg"]

    class_count_list = count_objects_in_images(yolo_model, image_paths)
    print(class_count_list)