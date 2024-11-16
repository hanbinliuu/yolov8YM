import cv2
import math
import numpy as np
from PIL import Image, ImageEnhance
from rapidocr_onnxruntime import RapidOCR, VisRes


class ImageProcessor:

    """ 同轴光环境 """

    def __init__(self, img_path):

        self.img_path = img_path
        self.PI = math.pi
        self.HEIGHT, self.WIDTH = None, None
        self.line_height_param = 3.5
        self.contrast_factor = 1.1

    @property
    def find_circle(self):

        # 预处理
        image = cv2.imread(self.img_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=80,
            param1=200,
            param2=200,
            minRadius=100,
            maxRadius=0
        )

        if circles is None:
            raise ValueError("No circles detected")

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

        if closest_circle is None:
            raise ValueError("No circles detected")

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
        return cropped_image, radius, center

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
        cv2.imwrite('/Users/hanbinliu/Desktop/to_rectangle2.png', output)

        return output

    def denoise_image(self, input_image, denoise_path):
        enhancer = ImageEnhance.Contrast(Image.fromarray(input_image))
        output_image = enhancer.enhance(self.contrast_factor)
        denoised_image = cv2.GaussianBlur(np.array(output_image), (1, 1), 0)
        cv2.imwrite(denoise_path, denoised_image)

        return denoised_image

    def rapidocr_detect(self, detect):
        engine = RapidOCR()
        result, elapse = engine(detect, det=True, rec=True)
        return result

    def rapidocr_detect_vis(self, detect):
        engine = RapidOCR()
        vis = VisRes(font_path="/Users/hanbinliu/Downloads/FZYTK.TTF")
        result, elapse_list = engine(detect)
        boxes, txts, scores = list(zip(*result))

        res = vis(detect, boxes, txts, scores)
        cv2.imwrite("vis_det_rec.png", res)

        return result

    def process_image(self, denoise_path):
        image, radius, center = self.find_circle
        output = self.coordination_transform(image)
        denoised_image = self.denoise_image(output, denoise_path)
        result = self.rapidocr_detect_vis(denoised_image)
        print(result[0][1])



if __name__ == '__main__':

    img_path = '/Users/hanbinliu/Desktop/WechatIMG28.jpg'
    denoise_path = '/Users/hanbinliu/Desktop/denoised2.png'
    processor = ImageProcessor(img_path)
    processor.process_image(denoise_path)