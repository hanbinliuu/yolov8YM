import cv2
import numpy as np

def find_closest_circle(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=300, param1=500, param2=30, minRadius=300)

    # circles = cv2.HoughCircles(
    #     gray,
    #     cv2.HOUGH_GRADIENT,
    #     dp=1,  # 累加器分辨率与图像分辨率的比值（1表示与图像分辨率相同）
    #     minDist=80,  # 检测到的圆之间的最小距离
    #     param1=200,  # Canny边缘检测的高阈值
    #     param2=200,  # 累加器阈值，越小越多圆，但可能会有假阳性
    #     minRadius=100,  # 最小圆半径
    #     maxRadius=0  # 最大圆半径（如果设置为0，函数将找到所有圆）
    # )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        height, width = gray.shape
        center_x, center_y = width // 2, height // 2
        closest_circle = None
        min_distance = float('inf')

        for circle in circles[0, :]:
            circle_center = (circle[0], circle[1])
            circle_radius = circle[2]
            distance = np.sqrt((center_x - circle_center[0]) ** 2 + (center_y - circle_center[1]) ** 2)

            if distance < min_distance:
                min_distance = distance
                closest_circle = (circle_center, circle_radius)

        if closest_circle is not None:
            center, radius = closest_circle

            # Create a circular mask
            mask = np.zeros_like(gray)
            cv2.circle(mask, center, radius, 255, thickness=-1)

            # Crop the image using the circular mask
            cropped_image = cv2.bitwise_and(image, image, mask=mask)

            # Calculate the size of the square region based on the diameter of the circle
            size = 2 * radius

            # Define the coordinates for the top-left and bottom-right corners of the square
            x1 = center[0] - radius
            y1 = center[1] - radius
            x2 = x1 + size
            y2 = y1 + size

            # Crop the image to the square region
            cropped_image = cropped_image[y1:y2, x1:x2]
            print(cropped_image.shape)
            cv2.imwrite('/Users/hanbinliu/Desktop/test_result.png', cropped_image)

            return cropped_image, radius, center
        else:
            print("No circles detected")
    else:
        print("No circles detected")


image_path = '/Users/hanbinliu/Desktop/test.png'
result = find_closest_circle(image_path)

if result:
    image, radius, center = result
    print("radius:", radius)
    print("Center:", center)
