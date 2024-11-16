from ultralytics import YOLO


model = YOLO('./mod/yolov8n.pt')
# model = YOLO('yolov8s-p2.yaml').load('./mod/yolov8s.pt')
if __name__ == '__main__':
    model.train(data=r'D:\pycharmProject\yolov8\lhb\v8train\datas\line.yaml', epochs=400,
                imgsz=640,
                workers=3, device='0')