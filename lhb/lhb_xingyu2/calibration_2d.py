import numpy as np
import cv2
import os


class CameraCalibration:
    def __init__(self, cornersnum_x, cornersnum_y, gridsize):
        self.cornersnum_x = cornersnum_x  # 标定板X方向角点数目
        self.cornersnum_y = cornersnum_y  # 标定板Y方向角点数目
        self.gridsize = gridsize  #
        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def cameracalibration_interparam(self, calibimgdir):
        """
        # 内参标定
        :param calibimgdir: 标定板图像地址，一般需要15-20张在视野不同位置的标定板图像
        :return:
        mtx：内参矩阵
        distortion：畸变系数
        avg_error：反投影误差
        """

        # 棋盘格模板规格,这是棋盘上每一行和每一列的角点数
        # w=棋盘板一行上黑白块的数量-1，h=棋盘板一列上黑白块的数量-1，例如：10x6的棋盘板，则(w,h)=(9,5)
        w = self.cornersnum_x
        h = self.cornersnum_y
        # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        objp = np.zeros((w * h, 3), np.float32)
        # 将世界坐标系建在标定板上，Z坐标为0，只需赋值x和y
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

        # 储存棋盘格角点的世界坐标和图像坐标对
        objpoints = []  # 在世界坐标系中的三维点
        imgpoints = []  # 在图像平面的二维点

        assert len(os.listdir(calibimgdir)) != 0

        idx = 0
        for fname in os.listdir(calibimgdir):
            idx += 1
            img = cv2.imread(os.path.join(calibimgdir, fname), 0)
            img = cv2.medianBlur(img, 5)
            # 找到棋盘格角点
            ret, corners = cv2.findChessboardCorners(img, (w, h), None)
            # 如果找到足够点对，将其存储起来
            if ret:
                objpoints.append(objp)
                # 得到更为准确的角点亚像素坐标
                corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), self.criteria)
                imgpoints.append(corners2)
                # # 将角点在图像上显示
                # w_Img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # cv2.drawChessboardCorners(w_Img, (w, h), corners2, ret)
                # cv2.imwrite("../../../test/0Corners" + str(idx) + ".png", w_Img)

        assert len(imgpoints) != 0

        # 标定
        # mtx: 内参矩阵； distortion: 畸变系数； rvecs: 旋转向量； tvecs: 平移向量；
        # mtx = [ fx  0   cx
        #         0   fy  cy
        #         0   0   1  ]
        ret, mtx, distortion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)

        # 反投影误差，评估结果的好坏，越接近0，说明结果越理想。
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, distortion)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        avg_error = total_error / len(objpoints)

        return mtx, distortion, avg_error

    def cameracalibration_exterparam(self, calibimg, mtx, distortion):
        """
        相机外参的标定。
        :param calibimg: 标定板图像
        :param mtx: 内参矩阵
        :param distortion: 畸变系数
        :return:
        rmatrix：旋转矩阵
        tvec：平移向量
        s：图像坐标系到世界坐标系转换需要的参数
        """
        # 棋盘格模板规格,这是棋盘上每一行和每一列的角点数
        w = self.cornersnum_x
        h = self.cornersnum_y
        # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        world_point = np.zeros((w * h, 3), np.float32)
        # 将世界坐标系建在标定板上，Z坐标为0，只需赋值x和y
        world_point[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        world_point[:, :2] = world_point[:, :2] * self.gridsize

        if len(calibimg.shape) != 2:
            calibimg = cv2.cvtColor(calibimg, cv2.COLOR_BGR2GRAY)

        calibimg = cv2.medianBlur(calibimg, 5)
        ret, corners = cv2.findChessboardCorners(calibimg, (w, h), None)
        if ret:
            img_point = cv2.cornerSubPix(calibimg, corners, (11, 11), (-1, -1), self.criteria)
            # 已知空间点的真实坐标和图像坐标，获得相机的位姿
            # 坐标点从世界坐标系到相机坐标系
            _, rvec, tvec, _ = cv2.solvePnPRansac(world_point, img_point, mtx, distortion)
            rmatrix = cv2.Rodrigues(rvec)[0]

            return rmatrix, rvec, tvec

    @staticmethod
    def eliminate_distortion(img, mtx, dist):
        """
        根据标定得到的相机内参矩阵和畸变系数去畸变。
        :param img: 图像数据。
        :param mtx: 内参矩阵。
        :param dist: 畸变系数。
        :return:
        dst：校正后的图像。
        """

        # 去畸变
        h, w = img.shape[:2]
        # 优化内参数和畸变系数，通过设定自由自由比例因子alpha。
        # 当alpha设为0的时候，将会返回一个剪裁过的将去畸变后不想要的像素去掉的内参数和畸变系数；
        # 当alpha设为1的时候，将会返回一个包含额外黑色像素点的内参数和畸变系数，并返回一个ROI用于将其剪裁掉。
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # 自由比例参数
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        # 根据前面ROI区域裁剪图片
        # x,y,w,h = roi
        # dst = dst[y:y+h, x:x+w]
        # cv2.imwrite('calibresult.png', dst)
        return dst


if __name__ == '__main__':
    calibration = CameraCalibration(11, 8, 0.02)
    mtx, distortion, avg_error = calibration.cameracalibration_interparam(
        r"C:\Users\lhb\Desktop\50mm")
    temp = 1
