from ultralytics import YOLO

model = YOLO('./yolov8_nuts.pt')

if __name__ == '__main__':

    results = model.track(source=0, conf=0.1,
                          iou=0.9, show=True, save=True, save_txt=True,
                          save_crop=True, save_conf=True, name= 'nuts',
                          tracker="bytetrack.yaml", vid_stride=2)
