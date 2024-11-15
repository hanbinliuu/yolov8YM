from PIL import Image

def crop_to_match(source_image_path, target_image_path, output_image_path):
    # 打开源图片
    source_image = Image.open(source_image_path)
    source_size = source_image.size

    # 打开目标图片
    target_image = Image.open(target_image_path)

    # 裁剪目标图片
    cropped_image = target_image.crop((0, 0, source_size[0], source_size[1]))

    # 保存裁剪后的图片
    cropped_image.save(output_image_path)

# 示例用法

p2_path = '/Users/hanbinliu/Desktop/data/pix-img/raw/已移除背景的IMG_1184.jpg'
p1_path = '/Users/hanbinliu/Desktop/data/pix-img/raw/已移除背景的IMG_1182.jpg'
output_path = '/Users/hanbinliu/Desktop/crop3.jpg'
crop_to_match(p1_path, p2_path, output_path)
