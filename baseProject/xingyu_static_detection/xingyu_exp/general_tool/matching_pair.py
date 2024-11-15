import numpy as np
import cv2
import time
import copy
import os
import sys


# 0. 图像预处理
def preprocess(img):
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_BIN = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_Smooth = cv2.GaussianBlur(img, (5, 5), 0)
    # img_Smooth = cv2.dilate(img_Smooth, None, iterations=3)
    img_Smooth = cv2.erode(img_Smooth, None, iterations=1)
    return img_Smooth


# 1.基于BFMatching 的SIFT
def SIFT_BFMatching(img1, img2):
    sift=cv2.xfeatures2d.SIFT_create()

    start = time.time()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    time_spend = time.time() - start
    print(time_spend)
    img1_kp = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))
    img2_kp = cv2.drawKeypoints(img2, kp2, img2, color=(255, 255, 0))
    hmerge = np.hstack((img1_kp, img2_kp))  ### 关键点图像

    # BFMatching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # ratio
    good = []
    goodMatch = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])
            goodMatch.append(m)
    img_knn = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)  ## knn匹配图像
    img_goodknn = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)  ## 筛选后匹配图像
    # cv2.imshow('img',img_goodknn)
    # cv2.waitKey(0)

    return kp1, kp2, goodMatch, hmerge, img_knn, img_goodknn


# 2.根据关键点计算转换矩阵
def compute_affineMatrix(kp1, kp2, matches):
    ptsA = [kp1[m.queryIdx].pt for m in matches]
    ptsB = [kp2[m.trainIdx].pt for m in matches]
    print(len(ptsA), len(ptsB))
    ptsA = np.float32(ptsA)
    ptsB = np.float32(ptsB)
    affine_matrix, flags = cv2.estimateAffinePartial2D(ptsA, ptsB)
    print("affine_matrix: ", affine_matrix)

    return affine_matrix


# 3.根据转换矩阵,处理图片
def transform(img, matrix):
    rows, cols, channels = img.shape

    dst = cv2.warpAffine(img, matrix, (cols, rows))
    return dst


def main(img_raw, img_target,):
    # 预处理
    new_img1 = preprocess(img_raw)
    new_img2 = preprocess(img_target)
    # new_img1 = img_raw
    # new_img2 = img_target

    dst1_input = copy.deepcopy(new_img1)
    dst2_input = copy.deepcopy(new_img2)
    # 提取关键点并匹配
    kp1, kp2, goodMatch, hmerge, img_knn, img_goodknn = SIFT_BFMatching(dst1_input, dst2_input)
    # 计算关键点
    affine_matrix = compute_affineMatrix(kp1, kp2, goodMatch)
    transform_img = transform(img_raw, affine_matrix)
    cv2.imwrite("/Users/hanbinliu/desktop/resample.png", transform_img)
    return transform_img


def fill_contour(image, output_path):
    # 将图像转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 寻找轮廓
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 根据轮廓的面积进行排序，选择面积最大的轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    # 创建一个空白图像
    filled_image = np.zeros_like(image)
    # 绘制填充轮廓
    cv2.drawContours(filled_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    # 保存填充后的图像
    cv2.imwrite(output_path, filled_image)


if __name__ == '__main__':
    # img_raw = "/Users/hanbinliu/Downloads/已移除背景的IMG_0929.png"
    # img_target = "/Users/hanbinliu/Downloads/已移除背景的IMG_0930.png"
    #
    # img_raw = cv2.imread(img_raw)
    # img_target = cv2.imread(img_target)
    #
    # fill_contour(img_raw,"/Users/hanbinliu/desktop/raw2.png")
    # fill_contour(img_target,"/Users/hanbinliu/desktop/target2.png")

    img_raw2 = "/Users/hanbinliu/Downloads/已移除背景的IMG_1340.png"
    img_target2 = "/Users/hanbinliu/Downloads/已移除背景的IMG_1337.png"
    img_raw2 = cv2.imread(img_raw2)
    img_target2 = cv2.imread(img_target2)

    main(img_raw=img_raw2, img_target=img_target2)
