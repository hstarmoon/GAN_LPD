import sys
import cv2, PIL, time
import glob
import numpy as np
import matplotlib.pyplot as plt
import random, string
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageOps, ImageFile
# import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os.path, shutil
import bezier
# sys.path.append("./LP_module.py")
# import Automold
from Automold import add_sun_flare, darken
from LP_module import LP_Module, France_LP_Module, UK_LP_Module, German_LP_Module, California_LP_Module, Vietnam_LP_Module, Taiwan_LP_Module
from image_process import Random_Process, Image_Process
import imgaug as ia
import imgaug.augmenters as iaa


print(type(LP_Module))
LP_M = LP_Module()
LP_M_CA = California_LP_Module()
LP_M_D = German_LP_Module()
LP_M_UK = UK_LP_Module()
LP_M_VN = Vietnam_LP_Module()
LP_M_T = Taiwan_LP_Module()
class License_Plate_GAN():

    def __init__(self):
        self.num_GAN_file = 1500
        self.switch_cyclegan = False
        self.num_distortion = 2
        self.count_img = -1
        self.datagen = ImageDataGenerator(
            # zca_whitening=True,
            # rotation_range=1,
            # shear_range=3.0, #旋轉拉伸角度
            # width_shift_range=0.05,
            # height_shift_range=0.05,
            channel_shift_range=130,
            brightness_range=[0.1, 0.2],
            # zoom_range=0.2,
            # horizontal_flip=True, #水平翻轉
            rescale=1 / 255,
            fill_mode='constant')
        self.datagen_backup = ImageDataGenerator(
            # zca_whitening=True,
            # rotation_range=1,
            shear_range=3.0,
            width_shift_range=0.05,
            height_shift_range=0.05,
            channel_shift_range=180.0,
            brightness_range=[-0.9, -0.9],
            # zoom_range=0.2,
            # horizontal_flip=True, #水平翻轉
            rescale=1 / 255,
            fill_mode='constant')

    def visualize(self, original, augmented):
      fig = plt.figure()
      plt.subplot(1,2,1)
      plt.title('Original image')
      plt.imshow(original)

      plt.subplot(1,2,2)
      plt.title('Augmented image')
      plt.imshow(augmented)

    def data_augmentation_tf(self, img):
        img_h = img.shape[0]
        img_w = img.shape[1]
        random_brightness = tf.image.random_brightness(img,max_delta=20)
        crop_img = tf.random_crop(img, [img_h-10, img_w-10, 3])
        # random_brightness = np.array(random_brightness)
        sess = tf.InteractiveSession()
        crop_img = cv2.cvtColor(crop_img.eval(), cv2.COLOR_BGR2RGB)
        # plt.imshow(crop_img)
        # plt.show()
        return crop_img
    def check_hist_max(self, index):
        if index > 255 or index < 0:
            return False
        else:
            return True
    def check_darken_img(self, img):
        img_h = img.shape[0]
        img_w = img.shape[1]
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        max_gray0 = np.max(hist)
        elem1 = np.argmax(hist[0:])
        max_gray1 = float(hist[elem1])
        elem2 = np.argmax(hist[1:])
        max_gray2 = float(hist[elem2])
        overflow = 1
        for index in range(1, 6):
            if self.check_hist_max(elem1+index):
                max_gray1 += float(hist[elem1+index])
            else:
                max_gray1 += float(hist[elem1 - 5 - overflow])
                overflow += 1
            if self.check_hist_max(elem1 - index):
                max_gray1 += float(hist[elem1 - index])
            else:
                max_gray1 += float(hist[elem1 + 5 + overflow])
                overflow += 1
        # max_index = hist.index(max_gray0)
        if max_gray1 > img_h*img_w*0.6:
            # cv2.namedWindow("test", 0)
            # cv2.imshow("test", img)
            # cv2.waitKey(0)
            return False
        else:
            return True
    def check_img(self, img):

        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        max_gray0 = np.max(hist)
        if max_gray0 > 10000:
            return False
        else:
            return True

    def generator_all(self):
        print("====================GAN LP ====================")
        out_dir = r"D:\data\Taiwan\military\GAN"
        start_GAN_LP = time.time()
        for index in range(self.num_GAN_file):
            # img_GAN, filename_GAN = LP_M_F.generator_use_for_train()

            img_GAN, filename_GAN = LP_M_T.generator_all()
            # img_GAN = Random_Process().add_gaussian_noise(img_GAN, 1)
            # cv2.namedWindow("test", 0)
            # cv2.imshow("test", img_GAN)
            # cv2.waitKey(0)

            # img_GAN = cv2.cvtColor(img_GAN, cv2.COLOR_RGB2GRAY)
            # img_GAN = Random_Process().random_distortion(img_GAN, shift_pixel=10)
            img_GAN = Random_Process().random_resize(img_GAN, 80, 127)
            img_GAN = Random_Process().random_blur(img_GAN, degree_range=(2,10))
            img_GAN = cv2.resize(img_GAN, (209, 88))

            # img_GAN = cv2.cvtColor(img_GAN, cv2.COLOR_RGB2GRAY)
            switch_darken = random.randint(0, 3)
            # if not self.switch_cyclegan:
            if switch_darken == 77:
                # img_GAN = cv2.cvtColor(img_GAN, cv2.COLOR_GRAY2RGB)
                img_GAN = darken(img_GAN)
                # img_GAN = cv2.cvtColor(img_GAN, cv2.COLOR_RGB2GRAY)
            # if not self.check_darken_img(img_GAN):
            #     continue
            # img_GAN = Random_Process().random_blur(img_GAN)


            result_path =  os.path.join(out_dir, r"{}.jpg".format(filename_GAN))
            if self.switch_cyclegan:
                img_GAN = Image_Process().image_padding(img_GAN, 256, 256, [0, 0, 0])
            cv2.imwrite(result_path, img_GAN)
            switch_shadow = random.randint(1, 12)  # 8%
            if not self.switch_cyclegan:
                if switch_shadow == 1:
                    hist = cv2.calcHist([img_GAN], [0], None, [256], [0, 256])
                    elem = np.argmax(hist[0:])
                    mode = random.randint(0, 4)
                    add_rate_shadow = 0.9
                    if elem < 60:
                        add_rate_shadow = random.randint(20, 60) * 0.01
                    elif elem < 100:
                        add_rate_shadow = random.randint(20, 60) * 0.01
                    elif elem < 120:
                        add_rate_shadow = random.randint(20, 60) * 0.01
                    elif elem < 190:
                        add_rate_shadow = random.randint(50, 70) * 0.01
                    # distort_img = Random_Process().random_resize(img_GAN, 30, 127)
                    img_result = Random_Process().random_shadow_img(img_GAN, add_rate_shadow, 0, mode)
                    img_name = os.path.join(out_dir, r"{}_sh.jpg".format(filename_GAN))
                    cv2.imwrite(img_name, img_result)
                else:  # 91%
                    switch_sun = random.randint(1, 18)  # 91% * 5% = 5%
                    if switch_sun == 1:
                        print(index)
                        hist = cv2.calcHist([img_GAN], [0], None, [256], [0, 256])
                        elem = np.argmax(hist[0:])
                        add_rate = random.randint(10, 20) * 0.01
                        bios = 0
                        radius = random.randint(60, 90)
                        if elem < 60:
                            add_rate = random.randint(40, 60) * 0.01
                            bios = random.randint(0, 8)
                        elif elem < 100:
                            add_rate = random.randint(40, 60) * 0.01
                        elif elem < 120:
                            add_rate = random.randint(40, 50) * 0.01
                        elif elem < 190:
                            add_rate = random.randint(30, 40) * 0.01
                        # distort_img = Random_Process().random_resize(img_GAN, 30, 127)
                        img_result = Random_Process().random_sun_flare_img(img_GAN, add_rate, bios,
                                                                           no_of_flare_circles=0, src_radius=radius)
                        img_name = os.path.join(out_dir, r"{}_sh.jpg".format(filename_GAN))
                        cv2.imwrite(img_name, img_result)

        end_GAN_LP = time.time()
        print("Time_GAN: " + str(end_GAN_LP - start_GAN_LP) + " sec")
    def generator_all_D(self):
        print("====================GAN LP ====================")
        out_dir = r"D:\data\EVS\German\new_data\train_5_9_GAN30W"
        start_GAN_LP = time.time()
        for index in range(self.num_GAN_file):
            # img_GAN, filename_GAN = LP_M_F.generator_use_for_train()

            img_GAN, filename_GAN = LP_M_D.generator_all()
            switch_darken = random.randint(0, 3)
            if switch_darken == 0:
                darken_img = darken(img_GAN)
                darken_result = self.check_darken_img(darken_img)
                if darken_result == True:
                    img_GAN = darken_img

            img_GAN = cv2.cvtColor(img_GAN, cv2.COLOR_RGB2GRAY)
            if 'u' in filename_GAN:
                img_GAN = Random_Process().random_blur(img_GAN, degree_range=(2, 4))
            else:
                img_GAN = Random_Process().random_blur(img_GAN, degree_range=(2, 7))
            img_GAN = Random_Process().random_distortion(img_GAN, shift_pixel=10)

            result_path = os.path.join(out_dir, r"{}.jpg".format(filename_GAN))
            cv2.imwrite(result_path, img_GAN)
            switch_shadow = random.randint(1, 12)  # 8%
            if switch_shadow == 1:
                hist = cv2.calcHist([img_GAN], [0], None, [256], [0, 256])
                elem = np.argmax(hist[0:])
                mode = random.randint(0, 4)
                add_rate_shadow = 0.9
                if elem < 60:
                    add_rate_shadow = random.randint(20, 60) * 0.01
                elif elem < 100:
                    add_rate_shadow = random.randint(20, 60) * 0.01
                elif elem < 120:
                    add_rate_shadow = random.randint(20, 60) * 0.01
                elif elem < 190:
                    add_rate_shadow = random.randint(50, 70) * 0.01
                distort_img = Random_Process().random_resize(img_GAN, 30, 127)
                img_result = Random_Process().random_shadow_img(distort_img, add_rate_shadow, 0, mode)
                img_name = os.path.join(out_dir, r"{}_sh.jpg".format(filename_GAN))
                cv2.imwrite(img_name, img_result)
            else:  # 91%
                switch_sun = random.randint(1, 18)  # 91% * 5% = 5%
                if switch_sun == 1:
                    hist = cv2.calcHist([img_GAN], [0], None, [256], [0, 256])
                    elem = np.argmax(hist[0:])
                    add_rate = random.randint(10, 20) * 0.01
                    bios = 0
                    radius = random.randint(60, 90)
                    if elem < 60:
                        add_rate = random.randint(40, 60) * 0.01
                        bios = random.randint(0, 8)
                    elif elem < 100:
                        add_rate = random.randint(40, 60) * 0.01
                    elif elem < 120:
                        add_rate = random.randint(40, 50) * 0.01
                    elif elem < 190:
                        add_rate = random.randint(30, 40) * 0.01
                    distort_img = Random_Process().random_resize(img_GAN, 30, 127)
                    img_result = Random_Process().random_sun_flare_img(distort_img, add_rate, bios,
                                                                       no_of_flare_circles=0, src_radius=radius)
                    img_name = os.path.join(out_dir, r"{}_sh.jpg".format(filename_GAN))
                    cv2.imwrite(img_name, img_result)

        end_GAN_LP = time.time()
        print("Time_GAN: " + str(end_GAN_LP - start_GAN_LP) + " sec")

    def generator_for_cycleGAN(self):
        print("====================GAN LP ====================")
        out_dir = r"D:\data\data_4GAN\EVS\German\gan_result"
        start_GAN_LP = time.time()
        for index in range(self.num_GAN_file):
            # img_GAN, filename_GAN = LP_M_F.generator_use_for_train()

            img_GAN, filename_GAN = LP_M_D.generator_new()

            img_GAN = cv2.cvtColor(img_GAN, cv2.COLOR_RGB2GRAY)
            img_GAN = cv2.cvtColor(img_GAN, cv2.COLOR_GRAY2RGB)
            img_GAN = Random_Process().random_blur(img_GAN)
            img_GAN = Image_Process().image_padding(img_GAN, 256, 256, [0, 0, 0])
            result_path = os.path.join(out_dir, filename_GAN)
            cv2.imwrite(result_path, img_GAN)

        end_GAN_LP = time.time()
        print("Time_GAN: " + str(end_GAN_LP - start_GAN_LP) + " sec")

    def get_gan_result_img(self):
        file_path = r"D:\data\data_4GAN\LP_GAN\GAN_test_all"
        out_dir = r"D:\data\data_4GAN\LP_GAN\GAN_test_all_crop"
        files = glob.glob(os.path.join(file_path, "*.jpg"))
        y1 = 101
        x1 = 0
        w = 256
        h = 54
        for index, file in enumerate(files):
            basename = os.path.basename(file)
            img = cv2.imdecode(np.fromfile(file, dtype = np.uint8), 1)
            crop_img = img[y1:y1+h, x1:x1+w]
            distortion_img = Random_Process().random_distortion(crop_img, shift_pixel=10)
            img = cv2.resize(distortion_img, (94, 21))
            cv2.imwrite(os.path.join(out_dir, basename), img)
            # filename = r"{}\{}.jpg".format(out_dir, str(index))
            # cv2.imwrite(filename, crop_img)


    def batch_process(self):
        file_path = r"D:\RD\Project\CycleGan\result\9700"
        out_dir = r"D:\data\data_4GAN\EVS\France\9700_1"
        files = glob.glob("{}\*.jpeg".format(file_path))
        num_generate_img = 1
        if len(files) < 1:
            files = glob.glob("{}\*.jpg".format(file_path))
        for index, file in enumerate(files):
            base_name = os.path.basename(file)[:-4]
            img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 0)
            for index in range(num_generate_img):
                # result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # result_img = Image_Process.image_padding(img, 256, 256, [0, 0, 0])


                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                elem = np.argmax(hist[0:])
                print(elem)

                mode = random.randint(0, 4)
                add_rate_shadow = 0.9
                add_rate = random.randint(1, 2) * 0.1
                bios = 0
                radius = random.randint(60, 90)

                if elem < 60:
                    add_rate = random.randint(7, 8) * 0.1
                    bios = random.randint(0, 8)
                    add_rate_shadow = random.randint(2, 6)*0.1
                elif elem < 100:
                    add_rate = random.randint(6, 8) * 0.1
                    add_rate_shadow = random.randint(2, 4)*0.1
                elif elem < 120:
                    add_rate = random.randint(4, 5) * 0.1
                    add_rate_shadow = random.randint(4, 6)*0.1
                elif elem < 190:
                    add_rate = random.randint(3, 4) * 0.1
                    add_rate_shadow = random.randint(6, 8)*0.1
                # add_rate_shadow = 0.1
                print(add_rate_shadow)
                switch = random.randint(0,1)
                if switch ==0:
                    shadow_img = Random_Process.random_shadow_img(self, img, add_rate_shadow, 0, mode)
                    filename = r"{}\{}_{}_S.jpg".format(out_dir, base_name, index )
                    cv2.imwrite(filename, shadow_img)
                else:
                    result_img = Random_Process.random_sun_flare_img(self, img, add_rate, bios, no_of_flare_circles=0,src_radius=radius)
                    filename = r"{}\{}_{}.jpg".format(out_dir, base_name, index)
                    cv2.imwrite(filename, result_img)
                # result_img = cv2.resize(img, (94, 21))
                # cv2.namedWindow("result_img", 0)
                # cv2.imshow("result_img", shadow_img)
                #
                # cv2.waitKey(0)
                # filename = r"{}\{}_{}".format(out_dir, base_name, index)


                a = 0

    def batch_process2(self):
        file_path = r"D:\data\cut_problem\Test Picture"
        out_dir = r"D:\data\cut_problem\RGB"
        files = glob.glob("{}\*.jpeg".format(file_path))
        if len(files) < 1:
            files = glob.glob("{}\*.jpg".format(file_path))
        for index, file in enumerate(files):
            base_name = os.path.basename(file)[:-4]
            img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result_path = r"{}\{}.jpg".format(out_dir, base_name)
            cv2.imwrite(result_path, img)
            # img = cv2.resize(img, (256,110))
            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # img = Random_Process().random_blur(img)
            # img_padding = Image_Process().image_padding(img, 256, 256, [0, 0, 0])
            # result_path = r"{}\{}.jpg".format(out_dir, base_name)
            # cv2.imwrite(result_path, img_padding)

    def copy_for_testbed_with_txt(self):
        img_list_txt_name = r"D:\data\EVS\clean\test_clean_error_label.txt"
        source_dir = os.path.dirname(img_list_txt_name)
        out_dir = r"D:\data\EVS\test_clean_error_label"
        with open(img_list_txt_name, 'r') as txt_file:
            img_name_list = txt_file.readlines()
        for img_name in img_name_list:
            img_name = img_name[2:-5]
            img_name_short = os.path.basename(img_name)
            file_path = r"{}\{}".format(source_dir, img_name)
            new_file_path = r"{}\{}".format(out_dir, img_name_short)
            if os.path.exists(r"{}.jpg".format(new_file_path)):
                continue
            if os.path.exists(r"{}.jpg".format(file_path)):
                shutil.copyfile(r"{}.jpg".format(file_path), r"{}.jpg".format(new_file_path))
            if os.path.exists(r"{}.txt".format(file_path)):
                shutil.copyfile(r"{}.txt".format(file_path), r"{}.txt".format(new_file_path))
    def test(self):
        path = r"D:\data\data_4GAN\val\2015.jpg"
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images = [image, image, image, image, image, image, image, image, image, image]
        seq = iaa.Sequential([
            # iaa.Fliplr(0.5),  # horizontal flips
            iaa.Crop(percent=(0, 0.01)),  # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast((0.25, 1.8)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.5, 1.5), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-5, 5),
                shear=(-8, 8)
            )
        ], random_order=True)  # apply augmenters in random order
        for index in range(101):
            images_aug = seq(images=images)
            # ia.imshow(images_aug)
            ia.imshow(np.hstack(images_aug))
        print("Augmented:")

if __name__ == '__main__':
    License_Plate_GAN().generator_all()
    # License_Plate_GAN().generator_for_cycleGAN()
    # License_Plate_GAN().get_gan_result_img()
    # License_Plate_GAN().batch_process2()
    # License_Plate_GAN().test()
    # License_Plate_GAN().copy_for_testbed_with_txt()
    # License_Plate_GAN().generator_for_cycleGAN()