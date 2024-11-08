import cv2

img_path = r"C:\Users\lhb\Desktop\15_57_57.jpg"
# img_path = r"C:\Users\lhb\Desktop\test\Image_20240103161415795.bmp"
# Read image
img = cv2.imread(img_path)
# 创建一个窗口
cv2.namedWindow("image", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)  # 是否显示网格

# Select ROI
rect = cv2.selectROI("image", img, False, False)
(x, y, w, h) = rect
print("x,y,w,h ={},{},{},{} ".format(x, y, w, h))

center_x = x + w / 2
center_y = y + h / 2
print("矩形中心点坐标为：", (center_x, center_y))
