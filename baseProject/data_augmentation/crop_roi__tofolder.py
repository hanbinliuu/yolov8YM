import fnmatch
import os
import time
import uuid

from PIL import Image


def traverse_directory(path, paths=None):
    """遍历文件夹下的所有文件夹，并保存路径"""
    if paths is None:
        paths = []
    for root, dirs, files in os.walk(path):
        for subdir in dirs:
            path_to_subdir = os.path.join(root, subdir)
            # print(path_to_subdir)
            paths.append(path_to_subdir)
            traverse_directory(path_to_subdir, paths)
    return paths


def has_images(folder_path):
    subfolders = [f for f in os.scandir(folder_path) if f.is_dir()]

    if subfolders:
        # 存在子文件夹，返回 False
        return False
    else:
        return True


def generate_random_filename():
    # 根据uuid生成
    return str(uuid.uuid4())


def batch_crop_and_save(input_folder, output_folder):
    global x, y, w, h, x2, y2, w2, h2
    all_path = traverse_directory(input_folder)

    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图像文件
    for filename in all_path:

        if has_images(filename):
            # 10
            if filename == "C:\\Users\\baseProject\\Desktop\\xingyu_pic\\10\\R2TBR26228AA\\cam1":
                x, y, w, h = 601, 1535, 136, 87
                x2, y2, w2, h2 = 0, 0, 0, 0
            if filename == "C:\\Users\\baseProject\\Desktop\\xingyu_pic\\10\\R2TBR26228AA\\cam3":
                x, y, w, h = 2174, 1661, 142, 64
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == "C:\\Users\\baseProject\\Desktop\\xingyu_pic\\10\\R2TBR26228AA\\cam5":
                x, y, w, h = 1569, 975, 168, 77
                x2, y2, w2, h2 = 2337, 869, 176, 75

            if filename == "C:\\Users\\baseProject\\Desktop\\xingyu_pic\\10\\R2TBR26229AA\\cam1":
                x, y, w, h = 764, 1542, 136, 83
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == "C:\\Users\\baseProject\\Desktop\\xingyu_pic\\10\\R2TBR26229AA\\cam3":
                x, y, w, h = 2316, 1648, 117, 61
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == "C:\\Users\\baseProject\\Desktop\\xingyu_pic\\10\\R2TBR26229AA\\cam5":
                x, y, w, h = 2968, 835, 187, 65
                x2, y2, w2, h2 = 2147, 886, 193, 72

            # 20
            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\20\P2S9R26228AA\cam5':
                x, y, w, h = 2404, 876, 170, 60
                x2, y2, w2, h2 = 1466, 1009, 145, 63

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\20\P2S9R26229AA\cam5':
                x, y, w, h = 2075, 905, 189, 75
                x2, y2, w2, h2 = 3043, 828, 190, 65

            # 30
            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\30\P2S9R21458AA\cam1':
                x, y, w, h = 1382, 1533, 182, 137
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\30\P2S9R21458AA\cam3':
                x, y, w, h = 1554, 1652, 347, 133
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\30\P2S9R21458AA\cam5':
                x, y, w, h = 3451, 678, 190, 89
                x2, y2, w2, h2 = 2795, 804, 187, 79
            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\30\P2S9R21459AA\cam1':
                x, y, w, h = 1179, 1538, 431, 146
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\30\P2S9R21459AA\cam3':
                x, y, w, h = 1598, 1657, 157, 121
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\30\P2S9R21459AA\cam5':
                x, y, w, h = 1709, 922, 181, 77
                x2, y2, w2, h2 = 1025, 941, 212, 82

            # 40
            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\40\PP624813AA\cam1':
                x, y, w, h = 666, 1551, 130, 58
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\40\PP624813AA\cam3':
                x, y, w, h = 2283, 1652, 159, 53
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\40\PP624813AA\cam5':
                x, y, w, h = 1404, 977, 182, 150
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\40\PP624814AA\cam1':
                x, y, w, h = 586, 1549, 107, 55
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\40\PP624814AA\cam3':
                x, y, w, h = 2236, 1659, 153, 46
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\40\PP624814AA\cam5':
                x, y, w, h = 3136, 777, 192, 157
                x2, y2, w2, h2 = 0, 0, 0, 0

            # 50
            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\50\PP624739AA\cam1':
                x,y,w,h =1168,1552,795,183
                x2, y2, w2, h2 = 0, 0, 0, 0
            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\50\PP624739AA\cam3':
                x, y, w, h = 1250, 1680, 628, 137
                x2, y2, w2, h2 = 0, 0, 0, 0
            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\50\PP624740AA\cam1':
                x, y, w, h = 1271, 1526, 710, 218
                x2, y2, w2, h2 = 0, 0, 0, 0
            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\50\PP624740AA\cam3':
                x, y, w, h = 1222, 1642, 651, 229
                x2, y2, w2, h2 = 0, 0, 0, 0

            # 60
            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\60\PP6V214K58AA\cam1':
                x, y, w, h = 1310, 1545, 454, 216
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\60\PP6V214K58AA\cam3':
                x, y, w, h = 2408, 1647, 154, 83
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\60\PP6V214K58AA\cam5':
                x, y, w, h = 848, 1084, 213, 154
                x2, y2, w2, h2 = 0, 0, 0, 0
                # 12.28更新
                x, y, w, h = 862, 1074, 160, 169

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\60\PP6V214K59AA\cam1':
                x, y, w, h = 445, 1538, 147, 85
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\60\PP6V214K59AA\cam3':
                x, y, w, h = 1428, 1679, 466, 165
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\60\PP6V214K59AA\cam5':
                x, y, w, h = 3662, 774, 199, 151
                x2, y2, w2, h2 = 0, 0, 0, 0

            # 70
            if filename == "C:\\Users\\baseProject\\Desktop\\xingyu_pic\\70\\PP620236AA\\cam1":
                x, y, w, h = 1285, 1467, 481, 219
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == "C:\\Users\\baseProject\\Desktop\\xingyu_pic\\70\\PP620236AA\\cam3":
                x, y, w, h = 1911, 1501, 212, 209
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == "C:\\Users\\baseProject\\Desktop\\xingyu_pic\\70\\PP620236AA\\cam4":
                x, y, w, h = 88, 700, 1292, 1037
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == "C:\\Users\\baseProject\\Desktop\\xingyu_pic\\70\\PP620237AA\\cam1":
                x, y, w, h = 1314, 1470, 475, 231
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == "C:\\Users\\baseProject\\Desktop\\xingyu_pic\\70\\PP620237AA\\cam3":
                x, y, w, h = 1875, 1489, 229, 199
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == "C:\\Users\\baseProject\\Desktop\\xingyu_pic\\70\\PP620237AA\\cam4":
                x, y, w, h = 25, 432, 1430, 1120
                x2, y2, w2, h2 = 0, 0, 0, 0

            # 80
            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\80\8890275004\cam1':
                x, y, w, h = 1894, 1555, 643, 299
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\80\8890275004\cam3':
                x, y, w, h = 808, 1365, 158, 156
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\80\8890276214\cam1':
                x, y, w, h = 1949, 1584, 573, 287
                x2, y2, w2, h2 = 0, 0, 0, 0

            if filename == r'C:\Users\lhb\Desktop\xingyu_pic\80\8890276214\cam3':
                x, y, w, h = 800, 1377, 139, 122
                x2, y2, w2, h2 = 0, 0, 0, 0

            # 开始裁
            for img in os.listdir(filename):
                if img.endswith(('.jpg', '.jpeg', '.png')):
                    # 构建输入和输出文件的完整路径
                    output_path = os.path.join(output_folder, img)

                    # 打开原图
                    img_path = os.path.join(filename, img)
                    original_image = Image.open(img_path)

                    # 根据给定的坐标和尺寸截取图像
                    cropped_image = original_image.crop((x, y, x + w, y + h))
                    cropped_image.save(output_path)
                    # 如果有第二个ROI
                    if x2 != 0 and y2 != 0 and w2 != 0 and h2 != 0:
                        cropped_image2 = original_image.crop((x2, y2, x2 + w2, y2 + h2))
                        timestamp = str(int(time.time()))
                        # output_path2 = output_path.replace('.jpeg', f'{timestamp}.jpeg')
                        output_path2 = output_path.replace('.jpeg', f'{timestamp}_2.jpeg')
                        # output_filename = generate_random_filename() + '.jpeg'
                        # output_path2 = os.path.join(output_folder, output_filename)
                        cropped_image2.save(output_path2)


if __name__ == '__main__':
    # 输入：原始照片文件夹
    # 输出：裁剪后的照片文件夹
    batch_crop_and_save(r"C:\Users\lhb\Desktop\xingyu_pic",
                        r"C:\Users\lhb\Desktop\croptest"
                        )
