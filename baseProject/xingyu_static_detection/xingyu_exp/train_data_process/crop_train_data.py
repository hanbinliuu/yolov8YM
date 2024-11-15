import os
from PIL import Image


def batch_crop_and_save(input_folder, output_folder, x, y, w, h):
    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图像文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # _, filename = filename.split("._", 1)
            # 构建输入和输出文件的完整路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 打开原图
            original_image = Image.open(input_path)

            # 根据给定的坐标和尺寸截取图像
            cropped_image = original_image.crop((x, y, x + w, y + h))

            # 保存截取后的图像到新文件夹
            cropped_image.save(output_path)


batch_crop_and_save(r"D:\mvs",
                    r"C:\Users\Administrator\Desktop\xingyu_test\test",
                    x = 775, y = 1341, w = 141, h = 211
                    )
