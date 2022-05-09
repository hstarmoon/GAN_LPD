import sys
import cv2, PIL, time
import glob
import numpy as np
import matplotlib.pyplot as plt
import random, string
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageOps, ImageFile
# import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os.path
import bezier
from Automold import add_sun_flare

class Random_Process():
    def __init__(self):
        pass
    def random_resize(self, img, size_mini, size_max):
        img_h = img.shape[0]
        img_w = img.shape[1]
        random_percent = np.random.randint(size_mini, size_max) / 100
        new_size = (int(img_w*random_percent), int(img_h*random_percent))
        img_resize = cv2.resize(img, new_size)
        return img_resize

    def randomColor(self, image):
        """
        對影象進行顏色抖動
        :param image: PIL的影象image
        :return: 有顏色色差的影象image
        """
        random_factor = np.random.randint(0, 30) / 10.  # 隨機因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 調整影象的飽和度
        random_factor = np.random.randint(10, 11) / 10.  # 隨機因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 調整影象的亮度
        random_factor = np.random.randint(10, 11) / 10.  # 隨機因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 調整影象對比度
        random_factor = np.random.randint(0, 1) / 10.  # 隨機因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 調整影象銳度

    def random_blur(self, img, degree_range=(2, 8), angle_range=(360, 360)):
        degree = np.random.randint(degree_range[0], degree_range[1])
        if angle_range[0] != 360:
            angle = np.random.randint(angle_range[0], angle_range[1])
        else:
            angle = 360
        # 生成任意角度的運動模糊kernel的矩陣， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree, degree), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(img, -1, motion_blur_kernel)
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred

    def random_distortion(self, img, shift_pixel=15, BG_mode="cifar"):
        img_h = img.shape[0]
        img_w = img.shape[1]
        shift_x_list = [15, 15, 15 + img_w, 15 + img_w]
        shift_y_list = [15, 15 + img_h, 15 + img_h, 15]
        for index in range(4):
            shift_x_list[index] += np.random.randint(-shift_pixel, shift_pixel)
            shift_y_list[index] += np.random.randint(-shift_pixel, shift_pixel)
        if BG_mode == 'cifar':
            img_padding = Image_Process().make_border_cifar(img,  15, 15, 15, 15)
        else:
            img_padding = cv2.copyMakeBorder(img, 15, 15, 15, 15, cv2.BORDER_CONSTANT)
        pts1 = np.float32(
            [[shift_x_list[0], shift_y_list[0]],[shift_x_list[1], shift_y_list[1]],
             [shift_x_list[2], shift_y_list[2]],[shift_x_list[3], shift_y_list[3]]])
        pts2 = np.float32(
            [[0, 0], [0, img_h], [img_w, img_h], [img_w, 0]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        rect_img = cv2.warpPerspective(img_padding, matrix, (img_w, img_h))
        return rect_img
    def random_shadow_img(self, img, img_add_rate, img_add_bios, mode):
        img_h = img.shape[0]
        img_w = img.shape[1]
        left_point = random.randint(int(img_h * 0.3), int(img_h * 0.6))
        right_point = random.randint(int(img_h * 0.3), int(img_h * 0.6))
        if mode == 0:
            control_pointx0 = random.randint(int(img_w * 0.1), int(img_w * 0.5))
            control_pointx1 = random.randint(int(img_w * 0.6), int(img_w * 0.9))
            control_pointy0 = random.randint(-int(img_h * 0.1), int(img_h * 0.25))
            control_pointy1 = random.randint(-int(img_h * 0.1), int(img_h * 0.25))
            points_tmp = np.asfortranarray([[0, control_pointx0, control_pointx1, img_w],
                                            [left_point, left_point + control_pointy0, left_point + control_pointy1,
                                             right_point]])
        elif mode == 1:
            points_tmp = np.asfortranarray([[0, img_w], [left_point, right_point]])
        elif mode == 2:
            control_pointy0 = random.randint(-int(img_h * 0.1), int(img_h * 0.25))
            points_tmp = np.asfortranarray(
                [[0, img_w * 0.5, img_w], [left_point, left_point + control_pointy0, right_point]])
        elif mode == 3:
            control_pointx0 = random.randint(int(img_w * 0.2), int(img_w * 0.8))
            control_pointy0 = random.randint(-int(img_h * 0.1), int(img_h * 0.25))
            points_tmp = np.asfortranarray(
                [[0, control_pointx0, img_w], [left_point, left_point + control_pointy0, right_point]])
        elif mode == 4:
            control_pointx0 = random.randint(int(img_w * 0.1), int(img_w * 0.5))
            control_pointx1 = random.randint(int(img_w * 0.6), int(img_w * 0.9))
            control_pointy0 = random.randint(-int(img_h * 0.1), int(img_h * 0.15))
            control_pointy1 = random.randint(-int(img_h * 0.1), int(img_h * 0.15))
            points_tmp = np.asfortranarray([[0, control_pointx0, control_pointx1, img_w],
                                            [left_point, left_point + control_pointy0, left_point + control_pointy1,
                                             right_point]])

        curve = bezier.Curve.from_nodes(points_tmp)
        curve.plot(num_pts=256)
        ax = plt.gca()
        line = ax.lines[0]
        curve_list = []
        x = line.get_xdata()
        y = line.get_ydata()
        for x_tmp, y_tmp in zip(x, y):
            curve_list.append([int(x_tmp + 0.5), int(y_tmp + 0.5)])

        white_BG = np.zeros(img.shape, np.uint8) + 255
        points = np.array([[0, 0]] + curve_list + [[img_w, 0]])
        black_img = cv2.fillPoly(white_BG.copy(), [points], (0, 0, 0))
        ROI = cv2.bitwise_and(black_img, img)
        add_img = cv2.addWeighted(img, img_add_rate, ROI, 1 - img_add_rate, img_add_bios)
        plt.close()
        # cv2.namedWindow("black_img", 0)
        # cv2.imshow("black_img", black_img)
        # cv2.namedWindow("ROI", 0)
        # cv2.imshow("ROI", ROI)
        # cv2.namedWindow("add_img", 0)
        # cv2.imshow("add_img", add_img)
        # cv2.waitKey(0)
        return add_img
    def random_sun_flare_img(self, img, add_rate, bios, no_of_flare_circles=0, src_radius=80, ):
        img_sun_flare = add_sun_flare(img, no_of_flare_circles=0, src_radius=80)
        add_img = cv2.addWeighted(img, add_rate, img_sun_flare, 1 - add_rate, bios)
        return add_img
    def tf_augmentation(self):
        # TensorFlow. 'x' = A placeholder for an image.
        shape = [height, width, channels]
        x = tf.placeholder(dtype=tf.float32, shape=shape)
        flip_2 = tf.image.flip_up_down(x)
        flip_3 = tf.image.flip_left_right(x)
        flip_4 = tf.image.random_flip_up_down(x)
        flip_5 = tf.image.random_flip_left_right(x)

        # To rotate in any angle. In the example below, 'angles' is in radians
        shape = [batch, height, width, 3]
        y = tf.placeholder(dtype = tf.float32, shape = shape)
        rot_tf_180 = tf.contrib.image.rotate(y, angles=3.1415)

    def add_gaussian_noise(self, img, noise_sigma):
        """
        给图片添加高斯噪声
        image_in:输入图片
        noise_sigma：
        """
        temp_image = np.float64(np.copy(img))

        h, w, _ = temp_image.shape
        # 标准正态分布*noise_sigma
        noise = np.random.randn(h, w) * noise_sigma

        noisy_image = np.zeros(temp_image.shape, np.float64)
        if len(temp_image.shape) == 2:
            noisy_image = temp_image + noise
        else:
            noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
            noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
            noisy_image[:, :, 2] = temp_image[:, :, 2] + noise

        return noisy_image



class Image_Process():
    def make_border_cifar(self, img, up, down, left, right):
        img_h = img.shape[0]
        img_w = img.shape[1]

        path_backgound = r"D:\data\CIFAR10\test_background"
        file_background = r"{}\{}.jpg".format(path_backgound, str(random.randint(0, 9999)))
        img_background = cv2.imread(file_background, cv2.IMREAD_GRAYSCALE)
        img_background = cv2.resize(img_background, (img_w + left + right, img_h + up + down))
        img_background[up:up+img_h, left:left+img_w] = img
        # img_background.paste(img, (left, up))
        return img_background
    def image_padding(self, image, target_height=64, target_width=64, back_ground_color=[0,0,0]):
        '''
        :param image: 輸入圖片
        :param target_height: 補0後高度
        :param target_width: 補0後寬度
        :return: 邊緣補0後的圖
        '''
        height = image.shape[0]
        width = image.shape[1]
        padding_height = (target_height - height) // 2
        padding_width = (target_width - width) // 2
        padding_height = 0 if padding_height < 0 else padding_height
        padding_width = 0 if padding_width < 0 else padding_width
        image = cv2.copyMakeBorder(image, padding_height, padding_height, padding_width, padding_width,
                                   cv2.BORDER_CONSTANT, value=back_ground_color)
        image = cv2.resize(image, (target_width, target_height))
        return image
    def data_augmentation_keras(self, img, channel_shift_range, brightness_range):
        img_h = img.shape[0]
        img_w = img.shape[1]
        img_shape = img.shape
        img = img.reshape((1,) + img.shape)
        datagen = ImageDataGenerator(
            # zca_whitening=True,
            # rotation_range=1,
            # shear_range=3.0, #旋轉拉伸角度
            # width_shift_range=0.05,
            # height_shift_range=0.05,
            channel_shift_range=channel_shift_range,
            brightness_range=brightness_range,
            # zoom_range=0.2,
            # horizontal_flip=True, #水平翻轉
            rescale=1 / 255,
            fill_mode='constant')
        batch = datagen.flow(img, batch_size=1)
        for index in batch:
            img_augment = index[0]
            # img_augment = img_augment.astype('float32')
            img_augment = ((img_augment - img_augment.min()) * (1/(img_augment.max() - img_augment.min()) * 255)).astype('uint8')
            # plt.imshow(img_augment)
            # plt.show()
            # img_augment = cv2.cvtColor(img_augment, cv2.COLOR_RGB2BGR)
            # plt.imshow(img_augment)
            # plt.show()
            return img_augment
            break
        return img

def test(self):
    degree_range = (2, 10)
    TEST = degree_range[0]
    degree = np.random.randint(degree_range(0), degree_range(1))
    pass
if __name__ == '__main__':
    test()

