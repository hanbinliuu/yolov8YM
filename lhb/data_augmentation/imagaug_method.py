#!usr/bin/python
# -*- coding: utf-8 -*-
import cv2
from imgaug import augmenters as iaa
import os

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# 定义一组变换方法.
seq2 = iaa.Sequential([

    # 每个图片选择选择这其中的0到5种方法做变换，随机生成
    # 将Augmenter中的部分变换应用在图片处理上，而不是应用所有的Augmenter。例如：可以定义20种变换，但每次只选择其中的5个。但是不支持固定选择某一个Augmenter
    # 参数n: 从总的Augmenters中选择多少个。可以是一个int, tuple, list或者随机；random_order:是否每次顺序不一样。
    iaa.SomeOf((1, 2),
               [
                   iaa.Fliplr(0.5),  # 对50%的图片进行水平镜像翻转
                   iaa.Flipud(0.5),  # 对50%的图片进行垂直镜像翻转

                   # Convert some images into their superpixel representation,
                   # sample between 20 and 200 superpixels per image, but do
                   # not replace all superpixels with their average, only
                   # some of them (p_replace).
                   # 对batch中的一部分图片应用一部分Augmenters,剩下的图片应用另外的Augmenters。
                   sometimes(
                       iaa.Superpixels(
                           p_replace=(0, 1.0),
                           n_segments=(20, 200)
                       )
                   ),

                   # Blur each image with varying strength using
                   # gaussian blur (sigma between 0 and 3.0),
                   # average/uniform blur (kernel size between 2x2 and 7x7)
                   # median blur (kernel size between 3x3 and 11x11).
                   # 每次从一系列Augmenters中选择一个来变换。
                   # iaa.OneOf([
                   #     # 高斯扰动
                   #     iaa.GaussianBlur((0, 0.1)),
                   #     # 从最邻近像素中取均值来扰动。
                   #     iaa.AverageBlur(k=(2, 7)),
                   #     # 通过最近邻中位数来扰动。
                   #     iaa.MedianBlur(k=(3, 5)),
                   # ]),

                   # Sharpen each image, overlay the result with the original
                   # image using an alpha between 0 (no sharpening) and 1
                   # (full sharpening effect).
                   # 锐化
                   iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                   # Same as sharpen, but for an embossing effect.
                   # 浮雕效果
                   iaa.Emboss(alpha=(0, 0.5), strength=(0, 1.0)),

                   # Add gaussian noise to some images.
                   # In 50% of these cases, the noise is randomly sampled per
                   # channel and pixel.
                   # In the other 50% of all cases it is sampled once per
                   # pixel (i.e. brightness change).
                   # 添加高斯噪声。
                   # iaa.AdditiveGaussianNoise(
                   #     loc=0, scale=(0.0, 0.02 * 255)
                   # ),

                   # Invert each image's chanell with 5% probability.
                   # This sets each pixel value v to 255-v.
                   # iaa.Invert(0.05, per_channel=True),  # invert color channels

                   # Add a value of -10 to 10 to each pixel.
                   # iaa.Add((-1, 1), per_channel=0.5),

                   # Add random values between -40 and 40 to images, with each value being sampled per pixel:
                   # 按像素加。
                   # iaa.AddElementwise((-2, 2)),

                   # Change brightness of images (50-150% of original value).
                   # iaa.Multiply((0.5, 1.2)),

                   # Multiply each pixel with a random value between 0.5 and 1.5.
                   # 按像素值乘。
                   # iaa.MultiplyElementwise((0.5, 1.5)),

                   # Improve or worsen the contrast of images.
                   # 改变图像的对比度
                   # iaa.ContrastNormalization((0.5, 1.5)),
                   iaa.Affine(rotate=(-10, 10)),  # 旋转
                   # iaa.Resize({"height": 80, "width": 64})

               ],
               # do all of the above augmentations in random order 定义随机数
               random_order=True
               )

], random_order=True)  # apply augmenters in random order


seq = iaa.Sequential(
    [
        #
        # Apply the following augmenters to most images.
        #
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.1), # vertically flip 20% of all images

        # crop some of the images by 0-10% of their height/width
        # sometimes(iaa.Crop(percent=(0, 0.1))),

        # Apply affine transformations to some of the images
        # - scale to 80-120% of image height/width (each axis independently)
        # - translate by -20 to +20 relative to height/width (per axis)
        # - rotate by -45 to +45 degrees
        # - shear by -16 to +16 degrees
        # - order: use nearest neighbour or bilinear interpolation (fast)
        # - mode: use any available mode to fill newly created pixels
        #         see API or scikit-image for which modes are available
        # - cval: if the mode is constant, then use a random brightness
        #         for the newly created pixels (e.g. sometimes black,
        #         sometimes white)
        sometimes(iaa.Affine(
            scale={"x": (1, 1.2), "y": (1, 1.2)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-5, 5),
            shear=(-16, 16),
            order=[0, 1],
            cval=(0, 255),
        )),

        #
        # Execute 0 to 5 of the following (less important) augmenters per
        # image. Don't execute all of them, as that would often be way too
        # strong.
        #
        iaa.SomeOf((0, 1),
            [
                # Convert some images into their superpixel representation,
                # sample between 20 and 200 superpixels per image, but do
                # not replace all superpixels with their average, only
                # some of them (p_replace).
                sometimes(
                    iaa.Superpixels(
                        p_replace=(0, 1.0),
                        n_segments=(20, 200)
                    )
                ),

                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.5)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                # iaa.Resize({"height": 100, "width": 80}),


                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                # Same as sharpen, but for an embossing effect.
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                # Search in some images either for all edges or for
                # directed edges. These edges are then marked in a black
                # and white image and overlayed with the original image
                # using an alpha of 0 to 0.7.
                sometimes(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.5)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),

                # Add gaussian noise to some images.
                # In 50% of these cases, the noise is randomly sampled per
                # channel and pixel.
                # In the other 50% of all cases it is sampled once per
                # pixel (i.e. brightness change).
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.2
                ),

                # Either drop randomly 1 to 10% of all pixels (i.e. set
                # them to black) or drop them on an image with 2-5% percent
                # of the original size, leading to large dropped
                # rectangles.
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.01, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]),

                # Invert each image's channel with 5% probability.
                # This sets each pixel value v to 255-v.
                iaa.Invert(0.05, per_channel=True), # invert color channels

                # Add a value of -10 to 10 to each pixel.
                iaa.Add((-10, 10), per_channel=0.5),

                # Change brightness of images (50-150% of original value).
                iaa.Multiply((0.5, 1), per_channel=0.5),

                # Improve or worsen the contrast of images.
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

                # Convert each image to grayscale and then overlay the
                # result with the original with random alpha. I.e. remove
                # colors with varying strengths.
                iaa.Grayscale(alpha=(0.0, 1.0)),

                # In some images move pixels locally around (with random
                # strengths).
                sometimes(
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                ),

                # In some images distort local areas with varying strength.
                # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
            ],
            # do all of the above augmentations in random order
            random_order=True
        )
    ],
    # do all of the above augmentations in random order
    random_order=True
)


# 图片文件相关路径
path = 'C:/Users/lhb/Desktop/lightenhance/'
# 保存路径可以是不同的路径，但是需要先把保存地方的文件夹建好
savedpath = 'C:/Users/lhb/Desktop/test/'

imglist = []
filelist = os.listdir(path)

# 遍历要增强的文件夹，把所有的图片保存在imglist中
for item in filelist:
    img = cv2.imread(path + item)
    # print('item is ',item)
    # print('img is ',img)
    # images = load_batch(batch_idx)
    imglist.append(img)
# print('imglist is ' ,imglist)
print('all the picture have been appent to imglist')

# 对文件夹中的图片进行增强操作，循环100次
for count in range(4):
    images_aug = seq.augment_images(imglist)
    for index in range(len(images_aug)):
        filename = str(count) + str(index+1003) + '.jpg'
        # 保存图片
        cv2.imwrite(savedpath + filename, images_aug[index])
        print('image of count%s index%s has been writen' % (count, index))
