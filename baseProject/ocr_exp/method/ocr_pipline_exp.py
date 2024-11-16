import cv2
import math
import numpy as np
from PIL import Image, ImageEnhance
from rapidocr_onnxruntime import RapidOCR


def find_closest_circle(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=300, param1=500, param2=30, minRadius=300)

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

            mask = np.zeros_like(gray)
            cv2.circle(mask, center, radius, 255, thickness=-1)

            cropped_image = cv2.bitwise_and(image, image, mask=mask)
            size = 2 * radius
            x1 = center[0] - radius
            y1 = center[1] - radius
            x2 = x1 + size
            y2 = y1 + size

            cropped_image = cropped_image[y1:y2, x1:x2]
            # cv2.imwrite('/Users/hanbinliu/Desktop/test_result.png', cropped_image)
            return cropped_image, radius, center
        else:
            print("No circles detected")
    else:
        print("No circles detected")


# --------------极坐标转换------------
# ----- 全局参数
# PAI值
PI = math.pi
# 设置输入图像固定尺寸（必要）
HEIGHT, WIDTH = 500, 500
# 输入图像圆的半径，一般是宽高一半
CIRCLE_RADIUS = int(HEIGHT / 2)
# 圆心坐标
CIRCLE_CENTER = [HEIGHT / 2, WIDTH / 2]
# 极坐标转换后图像的高，可自己设置
LINE_HEIGHT = int(CIRCLE_RADIUS / 1.5)
# 极坐标转换后图像的宽，一般是原来圆形的周长
LINE_WIDTH = int(2 * CIRCLE_RADIUS * PI)


def create_line_image(img):
    # 建立展开后的图像
    line_image = np.zeros((LINE_HEIGHT, LINE_WIDTH, 3), dtype=np.uint8)
    # 按照圆的极坐标赋值
    for row in range(line_image.shape[0]):
        for col in range(line_image.shape[1]):
            # 角度，最后的-0.1是用于优化结果，可以自行调整
            theta = PI * 2 / LINE_WIDTH * (col + 1) + 5
            # 半径，减1防止超界
            rho = CIRCLE_RADIUS - row

            x = int(CIRCLE_CENTER[0] + rho * math.cos(theta) + 0.0)
            y = int(CIRCLE_CENTER[1] - rho * math.sin(theta) + 0.0)

            # 赋值
            line_image[row, col, :] = img[y, x, :]
    # 如果想改变输出图像方向，旋转就行了
    line_image = cv2.rotate(line_image, cv2.ROTATE_180)
    line_image = cv2.flip(line_image, 0)

    return line_image


def cordination_transform(img):

    # 读取图像
    if img is None:
        print("please check image path")
        return
    img = cv2.resize(img, (HEIGHT, WIDTH))

    # 展示原图
    output = create_line_image(img)
    # # 展示结果
    # cv2.imshow("dst", output)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    cv2.imwrite('/Users/hanbinliu/Desktop/test_rectangle.png', output)

    return output

# 后处理
def denoise_image(input_image, denoise_path):

    contrast_factor = 1.2
    # 对比度增强
    enhancer = ImageEnhance.Contrast(Image.fromarray(input_image))
    output_image = enhancer.enhance(contrast_factor)

    # 使用高斯滤波进行图像降噪
    denoised_image = cv2.GaussianBlur(np.array(output_image), (1, 1), 0)
    cv2.imwrite(denoise_path, denoised_image)

    return denoised_image

# ocr
def rapidocr_detect(detect_path):
    engine = RapidOCR()
    result, elapse = engine(detect_path, det=True, rec=True)
    return result


if __name__ == '__main__':
    img_path = '/Users/hanbinliu/Desktop/test.png'

    # 找圆
    image, radius, center = find_closest_circle(img_path)
    # 极坐标转换
    output = cordination_transform(image)
    denoise_path = '/Users/hanbinliu/Desktop/denoised.png'
    denoised_image = denoise_image(output, denoise_path)

    # ocr
    result = rapidocr_detect(denoise_path)
    print(result[0][1])
