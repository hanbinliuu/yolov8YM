import cv2
import numpy as np


class ImageSimilarityDetector:

    def __init__(self, image1_path, image2_path):
        self.image1_path = image1_path
        self.image2_path = image2_path

    def _calculate_histogram_similarity(self, hist1, hist2):
        # 计算直方图相关性
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        # 计算直方图差异
        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        return correlation, diff

    def histogram_similarity(self):
        image1 = cv2.imread(self.image1_path)
        image2 = cv2.imread(self.image2_path)
        orin_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        changed_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        hist1 = cv2.calcHist([orin_gray], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([changed_gray], [0], None, [256], [0, 256])

        correlation, diff = self._calculate_histogram_similarity(hist1, hist2)
        return correlation, diff

    def _calculate_cosine_similarity(self, A, B):
        A = np.array(A)
        B = np.array(B)
        dot_product = np.dot(A, B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        return dot_product / (norm_A * norm_B)

    def contour_similarity(self):
        coordinates1 = self.find_and_draw_contours(self.image1_path)
        coordinates2 = self.find_and_draw_contours(self.image2_path)

        # 找到最大长度
        max_len = max(len(coordinates1), len(coordinates2))

        # 使用 np.pad 将两个坐标列表填充到相同的长度
        coordinates1 = np.pad(coordinates1, (0, max_len - len(coordinates1)), mode='constant')
        coordinates2 = np.pad(coordinates2, (0, max_len - len(coordinates2)), mode='constant')

        similarity = self._calculate_cosine_similarity(coordinates1, coordinates2)
        return similarity

    @staticmethod
    def find_and_draw_contours(image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=40, threshold2=400)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coordinates = []
        for contour in contours:
            for point in contour:
                x, y = point[0]
                coordinates.append(y)
        return coordinates


if __name__ == '__main__':

    image_path1 = '/Users/hanbinliu/Desktop/data/pix-img/已移除背景的IMG_0925.jpg'
    image_path2 = '/Users/hanbinliu/Desktop/data/pix-img/已移除背景的IMG_0923.jpg'
    similarity_detector = ImageSimilarityDetector(image_path1, image_path2)
    correlation, diff = similarity_detector.histogram_similarity()
    print(f"Histogram Correlation: {correlation}, Histogram Difference: {diff}")

    contour_similarity = similarity_detector.contour_similarity()
    print(f"Contour Similarity: {contour_similarity}")

    temp = 1
