import cv2
import math
import numpy as np
from PIL import Image, ImageEnhance
from rapidocr_onnxruntime import RapidOCR, VisRes
from typing import Tuple


def padding_img(img: np.ndarray,
                padding_value: Tuple[int, int, int, int],
                padding_color: Tuple = (0, 0, 0)) -> np.ndarray:
    padded_img = cv2.copyMakeBorder(img,
                                    padding_value[0],
                                    padding_value[1],
                                    padding_value[2],
                                    padding_value[3],
                                    cv2.BORDER_CONSTANT,
                                    value=padding_color)
    return padded_img


class ImageProcessor:
    """ 零度环光环境下 """

    def __init__(self, img_path):

        self.img_path = img_path
        self.PI = math.pi
        self.HEIGHT, self.WIDTH = None, None
        # 零度环光场景下
        # self.line_height_param = 2
        # self.contrast_factor = 1.1
        # 同轴光或环光
        self.line_height_param = 1.8
        self.contrast_factor = 0.6

    @property
    def find_circle(self):

        image = cv2.imread(self.img_path)

        # 定义圆心和半径
        center_coordinates = (1190, 1000)
        radius = 580

        # 创建一个全黑的背景，尺寸与原图一样
        mask = np.zeros(image.shape, dtype="uint8")
        # 在背景中绘制一个圆
        cv2.circle(mask, center_coordinates, radius, (255, 255, 255), -1)  # 绘制白色的圆，-1表示填充整个圆

        # 使用掩码与原图结合，得到只有圆部分的图像
        result = cv2.bitwise_and(image, mask)

        size = 2 * radius
        x1 = center_coordinates[0] - radius
        y1 = center_coordinates[1] - radius
        x2 = x1 + size
        y2 = y1 + size

        cropped_image = result[y1:y2, x1:x2]

        cv2.imwrite('C:/Users/Administrator/Desktop/circle.png', cropped_image)
        # 显示结果图像
        cv2.imshow("Resized Image", cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return cropped_image, radius, center_coordinates

    def create_line_image(self, img):

        CIRCLE_RADIUS = int(self.HEIGHT / 2)
        CIRCLE_CENTER = [self.HEIGHT / 2, self.WIDTH / 2]
        LINE_HEIGHT = int(CIRCLE_RADIUS / self.line_height_param)
        LINE_WIDTH = int(2 * CIRCLE_RADIUS * self.PI)
        line_image = np.zeros((LINE_HEIGHT, LINE_WIDTH, 3), dtype=np.uint8)
        for row in range(line_image.shape[0]):
            for col in range(line_image.shape[1]):
                theta = self.PI * 2 / LINE_WIDTH * (col + 1) + 5
                rho = CIRCLE_RADIUS - row

                x = int(CIRCLE_CENTER[0] + rho * math.cos(theta) + 0.0)
                y = int(CIRCLE_CENTER[1] - rho * math.sin(theta) + 0.0)

                line_image[row, col, :] = img[y, x, :]
        line_image = cv2.rotate(line_image, cv2.ROTATE_180)
        line_image = cv2.flip(line_image, 0)

        return line_image

    def coordination_transform(self, img):

        self.HEIGHT, self.WIDTH = img.shape[0], img.shape[1]
        if img is None:
            print("please check image path")
            return
        img = cv2.resize(img, (self.HEIGHT, self.WIDTH))
        output = self.create_line_image(img)
        cv2.imwrite('C:/Users/Administrator/Desktop/to_rectangle.png', output)

        return output

    def denoise_image(self, input_image, denoise_path):
        enhancer = ImageEnhance.Contrast(Image.fromarray(input_image))
        output_image = enhancer.enhance(self.contrast_factor)
        denoised_image = cv2.GaussianBlur(np.array(output_image), (1, 1), 0)
        cv2.imwrite(denoise_path, denoised_image)

        return denoised_image

    def rapidocr_detect(self, detect):
        engine = RapidOCR()
        result, elapse = engine(detect)
        return result

    def rapidocr_detect_vis(self, detect):
        engine = RapidOCR()
        vis = VisRes(font_path="resources/fonts/FZYTK.TTF")
        result, elapse_list = engine(detect, use_det=True, use_cls=True, use_rec=True)
        print(result)
        boxes, txts, scores = list(zip(*result))

        res = vis(detect, boxes, txts, scores)
        cv2.imwrite("vis_det_rec.png", res)

        return result

    def process_image(self, denoise_path):
        image, radius, center = self.find_circle
        output = self.coordination_transform(image)
        denoised_image = self.denoise_image(output, denoise_path)
        padded_img = padding_img(denoised_image, (30, 30, 0, 0), (255, 255, 255))

        cv2.imshow('Binary Image', padded_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        result = self.rapidocr_detect(padded_img)

        print(result[0][1])


if __name__ == '__main__':
    # img_path = "C:/Users/Administrator/Desktop/20231117-130816.jpg"
    img_path = "C:/Users/Administrator/Desktop/20231117-131537.jpg"
    denoise_path = 'C:/Users/Administrator/Desktop/denoised.png'
    processor = ImageProcessor(img_path)
    processor.process_image(denoise_path)
