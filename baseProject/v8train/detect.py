from ultralytics import YOLO
import cv2

# model = YOLO(r"C:\Users\baseProject\Desktop\xingyu_test\small-object\best.pt")
model = YOLO(r"C:\Users\lhb\Desktop\best.pt")
img = cv2.imread(r"C:\Users\lhb\Desktop\test2\20240327145159440_cropped_0.jpeg")
results = model.predict(img, device='0', save=True)
