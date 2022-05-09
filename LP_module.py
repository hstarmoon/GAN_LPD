import random, string
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from keras.preprocessing.image import ImageDataGenerator
from image_process import Random_Process, Image_Process
from Automold import darken

class LP_Module(object):
    def __init__(self):
        self.out_dir = r"D:\data\Myanmar\Myanmar_ray\Myanmar.03\data\train_tiny_20_5_crop\gan"
        self.black_module_path = r"D:\data\data_4GAN\3615_ps.jpg"
        self.black1_module_path = r"D:\data\data_4GAN\black_one_S5066_ps.jpg"
        self.red_module_path = r"D:\data\data_4GAN\2R6615_ps.jpg"
        self.red1_module_path = r"D:\data\data_4GAN\S3844_ps.jpg"
        self.yellow_module_path = r"D:\data\data_4GAN\RLG3942_ps.jpg"
        self.HSE_black_module_path = r"D:\data\data_4GAN\HSE6020_ps.jpg"
        self.HSE_red_module_path = r"D:\data\data_4GAN\HSE3829_ps.jpg"
        self.UN_module_path = r"D:\data\data_4GAN\UN514_ps.jpg"
        self.fontPath = r"C:\Users\Albert\Desktop\font\Myanmar.ttf"
        self.enlish_word = 'ABCDEFGHIJKLMNPQ1234567890'
        self.font = ImageFont.truetype(self.fontPath, 74)
        black_ps = cv2.imdecode(np.fromfile(self.black_module_path, dtype=np.uint8), 3)
        self.black_ps = cv2.cvtColor(black_ps, cv2.COLOR_BGR2RGB)
        red_ps = cv2.imdecode(np.fromfile(self.red_module_path, dtype=np.uint8), 3)
        self.red_ps = cv2.cvtColor(red_ps, cv2.COLOR_BGR2RGB)
        black1_ps = cv2.imdecode(np.fromfile(self.black1_module_path, dtype=np.uint8), 3)
        self.black1_ps = cv2.cvtColor(black1_ps, cv2.COLOR_BGR2RGB)
        red1_ps = cv2.imdecode(np.fromfile(self.red1_module_path, dtype=np.uint8), 3)
        self.red1_ps = cv2.cvtColor(red1_ps, cv2.COLOR_BGR2RGB)
        yellow_ps = cv2.imdecode(np.fromfile(self.yellow_module_path, dtype=np.uint8), 3)
        self.yellow_ps = cv2.cvtColor(yellow_ps, cv2.COLOR_BGR2RGB)
        HSE_black_ps = cv2.imdecode(np.fromfile(self.HSE_black_module_path, dtype=np.uint8), 3)
        self.HSE_black_ps = cv2.cvtColor(HSE_black_ps, cv2.COLOR_BGR2RGB)
        HSE_red_ps = cv2.imdecode(np.fromfile(self.HSE_red_module_path, dtype=np.uint8), 3)
        self.HSE_red_ps = cv2.cvtColor(HSE_red_ps, cv2.COLOR_BGR2RGB)

        UN_img = cv2.imdecode(np.fromfile(self.UN_module_path, dtype=np.uint8), 3)
        self.UN_img = cv2.cvtColor(UN_img, cv2.COLOR_BGR2RGB)
    def generator_black(self, switch_Color):
        num_english_word = 2
        color_step = random.randint(-50, 50)
        colorA = 195 + color_step
        colorB = 195 + color_step
        colorC = 195 + color_step

        word_color = (colorA, colorB, colorC)
        img_copy = self.black_ps.copy()
        imgPil = Image.fromarray(img_copy)
        word_first = ''
        for index_word in range(num_english_word):
            word_tmp = ''.join(random.choices(self.enlish_word))
            word_first += word_tmp
        # word_first = ''.join(random.choices(string.ascii_uppercase + string.digits, k=2))
        word_last = ''.join(random.choices(string.digits, k=4))
        draw = ImageDraw.Draw(imgPil)
        # RGB
        # word_color = (206, 222, 238)
        draw.text((11, 22), word_first, font=self.font, fill=word_color)
        #(250, 232, 215)
        draw.text((86, 22), word_last, font=self.font, fill=word_color)
        img_copy = np.array(imgPil)
        if switch_Color == 1:
            img_copy = Image_Process().data_augmentation_keras(img_copy, 80, [0.1, 2.2])
        filename = r"{}\{}{}_GAN.jpg".format(self.out_dir, word_first, word_last)
        return img_copy, filename
    def generator_red(self, switch_Color):
        num_english_word = 2
        color_step = random.randint(-20, 50)
        colorA = 195 + color_step
        colorB = 195 + color_step
        colorC = 195 + color_step

        word_color = (colorA, colorB, colorC)
        img_copy = self.red_ps.copy()
        imgPil = Image.fromarray(img_copy)
        word_first = ''
        for index_word in range(num_english_word):
            word_tmp = ''.join(random.choices(self.enlish_word))
            word_first += word_tmp
        word_last = ''.join(random.choices(string.digits, k=4))
        draw = ImageDraw.Draw(imgPil)
        # RGB
        # word_color = (190, 190, 190)
        draw.text((11, 22), word_first, font=self.font, fill=word_color)
        #(250, 232, 215)
        draw.text((86, 22), word_last, font=self.font, fill=word_color)
        img_copy = np.array(imgPil)
        if switch_Color == 1:
            img_copy = Image_Process().data_augmentation_keras(img_copy, 80, [0.1, 0.2])
        filename = r"{}\{}{}_GAN.jpg".format(self.out_dir, word_first, word_last)
        return img_copy, filename
    def generator_black_oneword(self, switch_Color):
        num_english_word = 1
        enlish_one_word = 'IJKLMPQRSTX'
        img_copy = self.black1_ps.copy()
        imgPil = Image.fromarray(img_copy)
        word_first = ''
        for index_word in range(num_english_word):
            word_tmp = ''.join(random.choices(enlish_one_word))
            word_first += word_tmp
        word_last = ''.join(random.choices(string.digits, k=4))
        draw = ImageDraw.Draw(imgPil)
        # RGB
        word_color = (85, 94, 93)
        draw.text((30, 22), word_first, font=self.font, fill=word_color)
        draw.text((74, 22), word_last, font=self.font, fill=word_color)
        img_copy = np.array(imgPil)
        if switch_Color == 1:
            img_copy = Image_Process().data_augmentation_keras(img_copy, 30, [0.1,1.6])
        filename = r"{}\{}{}_GAN.jpg".format(self.out_dir, word_first, word_last)
        return img_copy, filename
    def generator_red_oneword(self, switch_Color):
        num_english_word = 1
        enlish_one_word = 'IJKLMPQRSTX'
        color_step = random.randint(-20, 50)
        colorA = 195 + color_step
        colorB = 195 + color_step
        colorC = 195 + color_step
        word_color = (colorA, colorB, colorC)
        img_copy = self.red1_ps.copy()
        imgPil = Image.fromarray(img_copy)
        word_first = ''
        for index_word in range(num_english_word):
            word_tmp = ''.join(random.choices(enlish_one_word))
            word_first += word_tmp
        word_last = ''.join(random.choices(string.digits, k=4))
        draw = ImageDraw.Draw(imgPil)
        draw.text((30, 22), word_first, font=self.font, fill=word_color)
        draw.text((74, 22), word_last, font=self.font, fill=word_color)
        img_copy = np.array(imgPil)
        if switch_Color == 1:
            img_copy = Image_Process().data_augmentation_keras(img_copy, 0, [0.1,1.0])
        filename = r"{}\{}{}_red_GAN.jpg".format(self.out_dir, word_first, word_last)
        return img_copy, filename
    def generator_yellow(self, switch_Color):
        img_copy = self.yellow_ps.copy()
        imgPil = Image.fromarray(img_copy)
        word_last = ''.join(random.choices(string.digits, k=4))
        draw = ImageDraw.Draw(imgPil)
        word_color = (142, 87, 7)
        draw.text((92, 22), word_last, font=self.font, fill=word_color)
        img_copy = np.array(imgPil)
        if switch_Color == 1:
            img_copy = Image_Process().data_augmentation_keras(img_copy, 130, [0.1, 0.2])
        word_first = 'RLG'
        filename = r"{}\{}{}_GAN.jpg".format(self.out_dir, word_first, word_last)
        return img_copy, filename
    def generator_HSE_black(self, switch_Color):
        img_copy = self.HSE_black_ps.copy()
        imgPil = Image.fromarray(img_copy)
        word_last = ''.join(random.choices(string.digits, k=4))
        draw = ImageDraw.Draw(imgPil)
        word_color = (107, 107, 107)
        draw.text((92, 22), word_last, font=self.font, fill=word_color)
        img_copy = np.array(imgPil)
        if switch_Color == 1:
            img_copy = Image_Process().data_augmentation_keras(img_copy, 50, [0.1, 1.2])
        word_first = 'HSE'
        filename = r"{}\{}{}_GAN.jpg".format(self.out_dir, word_first, word_last)
        return img_copy, filename
    def generator_HSE_red(self, switch_Color):
        img_copy = self.HSE_red_ps.copy()
        imgPil = Image.fromarray(img_copy)
        word_last = ''.join(random.choices(string.digits, k=4))
        draw = ImageDraw.Draw(imgPil)
        word_color = (201, 204, 193)
        draw.text((92, 22), word_last, font=self.font, fill=word_color)
        img_copy = np.array(imgPil)
        if switch_Color == 1:
            img_copy = Image_Process().data_augmentation_keras(img_copy, 50, [0.1, 1.9])
        word_first = 'HSE'
        filename = r"{}\{}{}_red_GAN.jpg".format(self.out_dir, word_first, word_last)
        return img_copy, filename
    def generator_UN(self, switch_Color):
        font = ImageFont.truetype(self.fontPath, 73)
        img_copy = self.UN_img.copy()
        imgPil = Image.fromarray(img_copy)
        word_first = ''.join(random.choices(string.digits, k=1))
        word_last = ''.join(random.choices(string.digits, k=2))
        draw = ImageDraw.Draw(imgPil)
        word_color = (26, 31, 37)
        draw.text((94, 22), word_first, font=font, fill=word_color)
        draw.text((145, 22), word_last, font=font, fill=word_color)
        img_copy = np.array(imgPil)
        if switch_Color == 1:
            img_copy = Image_Process().data_augmentation_keras(img_copy, 20, [0.1, 1.5])
        word_first = 'UN'
        filename = r"{}\{}{}_GAN.jpg".format(self.out_dir, word_first, word_last)
        return img_copy, filename
    def generator_UN_1000(self, switch_blur, switch_Color, switch_distortion, switch_padding, num_random):
        font = ImageFont.truetype(self.fontPath, 73)
        word_color = (26, 31, 37)
        for index in range(1000):
            img_copy = self.UN_img.copy()
            imgPil = Image.fromarray(img_copy)
            index_tmp = str(index).zfill(3)
            word_first = index_tmp[0:1]
            word_last = index_tmp[1:]
            draw = ImageDraw.Draw(imgPil)
            draw.text((94, 22), word_first, font=font, fill=word_color)
            #(250, 232, 215)
            draw.text((145, 22), word_last, font=font, fill=word_color)
            img_copy = np.array(imgPil)
            if switch_Color == 1:
                result_img = Image_Process().data_augmentation_keras(img_copy, 20, [0.1, 1.5])
            result_img_o = Random_Process().random_blur(result_img)
            filename = r"{}\UN{}{}_GAN_o.jpg".format(self.out_dir, word_first, word_last)
            cv2.imwrite(filename, result_img_o)
            for index in range(self.num_distortion):
                result_img2 = result_img
                if switch_blur == 1:
                    result_img2 = Random_Process().random_blur(result_img2)
                elif switch_blur == 2:
                    if index == num_random - 2:
                        result_img2 = Random_Process().random_blur(result_img2, 1)
                    else:
                        result_img2 = Random_Process().random_blur(result_img2)
                if switch_distortion == 1:
                    result_img2 = Random_Process().random_distortion(result_img2)
                elif switch_distortion == 2:
                    if index == num_random - 1:
                        result_img2 = Random_Process().random_distortion(result_img2, 1)
                    else:
                        result_img2 = Random_Process().random_distortion(result_img2)
                if switch_padding == 1:
                    result_img2 = Image_Process.image_padding(result_img2, 256, 256, [0, 0, 0])
                filename = r"{}\UN{}{}_GAN_{}.jpg".format(self.out_dir, word_first, word_last, str(index))
                cv2.imwrite(filename, result_img2)

import re


class France_LP_Module():
    def __init__(self):
        self.out_dir = r"D:\data\Myanmar\Myanmar_ray\Myanmar.03\data\train_tiny_20_5_crop\gan"
        self.new_module_path = r"D:\data\data_4GAN\EVS\France\new_ps.jpg"
        self.old_white_module_path = r""
        self.old_yellow_module_path = r""
        self.fontPath = r"C:\Users\Albert\Desktop\E_font\France.ttf"
        self.enlish_word = 'ABCDEFGHJKLMNPQRSTVWXYZ'
        self.enlish_word_use_for_train = 'ABBBBBBBCCCDDDDDDDEFGGGHJKLMMMMMMMNPQQQQQQQRSTVWWWWWWWXYZ'
        self.number_word_use_for_train = '00000012345678888889'
        self.font = ImageFont.truetype(self.fontPath, 46)
        self.font_country = ImageFont.truetype(self.fontPath, 21)
        self.new_ps = cv2.imdecode(np.fromfile(self.new_module_path, dtype=np.uint8), 3)
        # self.new_ps = cv2.cvtColor(new_ps, cv2.COLOR_BGR2RGB)
        # red_ps = cv2.imdecode(np.fromfile(self.red_module_path, dtype=np.uint8), 3)
        # self.red_ps = cv2.cvtColor(red_ps, cv2.COLOR_BGR2RGB)
    def get_country_number(self):
        num = random.randint(0,96)
        num = str(num).zfill(2)
        if num==20:
            num = '2A'
        elif num == 96:
            num = '2B'
        elif num == 97:
            num = '971'
        elif num == 98:
            num = '972'
        elif num == 99:
            num = '973'
        elif num == 100:
            num = '974'
        elif num == 101:
            num = '976'
        return num
    def generator_new(self):
        num_english_word = 2

        img_copy = self.new_ps.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''
        for index_word in range(num_english_word):
            word_tmp = ''.join(random.choices(self.enlish_word))
            word_1 += word_tmp
        word_3 = ''
        for index_word in range(num_english_word):
            word_tmp = ''.join(random.choices(self.enlish_word))
            word_3 += word_tmp

        draw = ImageDraw.Draw(imgPil)
        word_color = (0, 0, 0)
        draw.text((37, 11), word_1, font=self.font, fill=word_color)
        # (250, 232, 215)
        word_2 = ''.join(random.choices(string.digits, k=3))
        draw.text((112, 11), word_2, font=self.font, fill=word_color)
        draw.text((211, 11), word_3, font=self.font, fill=word_color)
        word_contry = self.get_country_number()
        word_color = (255, 255, 255)
        draw.text((273, 35), word_contry, font=self.font_country, fill=word_color)
        img_copy = np.array(imgPil)
        # cv2.namedWindow("test")
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}{}{}_GAN.jpg".format(word_1, word_2, word_3)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename
    def generator_use_for_train(self):
        num_english_word = 2
        img_copy = self.new_ps.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''
        for index_word in range(num_english_word):
            word_tmp = ''.join(random.choices(self.enlish_word_use_for_train))
            word_1 += word_tmp
        word_3 = ''
        for index_word in range(num_english_word):
            word_tmp = ''.join(random.choices(self.enlish_word_use_for_train))
            word_3 += word_tmp

        word_2 = ''
        for index_word in range(3):
            word_tmp = ''.join(random.choices(self.number_word_use_for_train))
            word_2 += word_tmp

        draw = ImageDraw.Draw(imgPil)
        word_color = (0, 0, 0)
        draw.text((37, 11), word_1, font=self.font, fill=word_color)
        # (250, 232, 215)
        # word_2 = ''.join(random.choices(string.digits, k=3))
        draw.text((112, 11), word_2, font=self.font, fill=word_color)
        draw.text((211, 11), word_3, font=self.font, fill=word_color)
        word_contry = self.get_country_number()
        word_color = (255, 255, 255)
        draw.text((273, 35), word_contry, font=self.font_country, fill=word_color)
        img_copy = np.array(imgPil)
        # cv2.namedWindow("test")
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}{}{}_GAN.jpg".format(word_1, word_2, word_3)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename

class UK_LP_Module():
    def __init__(self):
        self.out_dir = r"D:\data\data_4GAN\EVS\GER_\gan_result"
        self.new_module_path = r"D:\data\data_4GAN\EVS\UK\new_ps1.jpg"
        self.new_module_path2 = r"D:\data\data_4GAN\EVS\UK\new_ps2.jpg"
        self.old_white_module_path = r""
        self.old_yellow_module_path = r""
        self.fontPath = r"C:\Users\Albert\Desktop\E_font\UK1.ttf"
        # self.word_local_num_1 = 'ABCDEFGHJKLMNOPRSTUVWXY'
        # self.word_local_num_2 = 'ABCDEFGHJKLMNOPRSTUVWXY'
        self.enlish_word = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'
        self.font = ImageFont.truetype(self.fontPath, 59)
        self.font_country = ImageFont.truetype(self.fontPath, 21)
        self.new_ps = cv2.imdecode(np.fromfile(self.new_module_path, dtype=np.uint8), 3)
        self.new_ps2 = cv2.imdecode(np.fromfile(self.new_module_path2, dtype=np.uint8), 3)
    def generator_new1(self):
        num_english_word = 3

        img_copy = self.new_ps.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''
        word_1 += ''.join(random.choices(self.enlish_word))
        word_1 += ''.join(random.choices(self.enlish_word))
        word_2 = ''.join(random.choices(string.digits, k=2))
        word_3 = ''
        for index_word in range(num_english_word):
            word_tmp = ''.join(random.choices(self.enlish_word))
            word_3 += word_tmp

        draw = ImageDraw.Draw(imgPil)
        word_color = (0, 0, 0)
        label = word_1 + word_2 + word_3
        word_1 = word_1 + word_2 + '-' + word_3
        # word_1 = "BD51-SMR"
        draw.text((30, 7), word_1, font=self.font, fill=word_color)
        # (250, 232, 215)
        # draw.text((112, 11), word_2, font=self.font, fill=word_color)
        # draw.text((153, 7), word_3, font=self.font, fill=word_color)
        # word_contry = self.get_country_number()
        # word_color = (255, 255, 255)
        # draw.text((273, 35), word_contry, font=self.font_country, fill=word_color)
        img_copy = np.array(imgPil)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image_Process().data_augmentation_keras(img_copy, 100, [0.1, 2.2])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename
    def generator_new2(self):
        num_english_word = 3

        img_copy = self.new_ps2.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''
        word_1 += ''.join(random.choices(self.enlish_word))
        word_1 += ''.join(random.choices(self.enlish_word))
        word_2 = ''.join(random.choices(string.digits, k=2))
        word_3 = ''
        for index_word in range(num_english_word):
            word_tmp = ''.join(random.choices(self.enlish_word))
            word_3 += word_tmp

        draw = ImageDraw.Draw(imgPil)
        word_color = (0, 0, 0)
        label = word_1 + word_2 + word_3
        word_1 = word_1 + word_2 + '-' + word_3
        if '1' in word_2:
            draw.text((25, 8), word_1, font=self.font, fill=word_color)
        else:
            draw.text((19, 8), word_1, font=self.font, fill=word_color)
        # draw.text((29, 7), word_1, font=self.font, fill=word_color)
        # (250, 232, 215)
        # draw.text((112, 11), word_2, font=self.font, fill=word_color)
        # draw.text((153, 7), word_3, font=self.font, fill=word_color)
        # word_contry = self.get_country_number()
        # word_color = (255, 255, 255)
        # draw.text((273, 35), word_contry, font=self.font_country, fill=word_color)
        img_copy = np.array(imgPil)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image_Process().data_augmentation_keras(img_copy, 100, [0.1, 2.2])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename

    def generator_all(self):
        switch = random.randint(0,1)
        if switch ==1:
            A, B = self.generator_new1()
        else:
            A, B = self.generator_new2()
        return A, B


class German_LP_Module():
    def __init__(self):
        self.out_dir = r"D:\data\data_4GAN\EVS\German\gan_result"
        self.new_module_path1 = r"D:\data\data_4GAN\EVS\German\new_ps11.jpg"
        self.new_module_path2 = r"D:\data\data_4GAN\EVS\German\new_ps2.jpg"
        self.new_module_path3 = r"D:\data\data_4GAN\EVS\German\new_ps3.jpg"
        self.mark_path = r"D:\data\data_4GAN\EVS\German\mark_ps.jpg"
        self.old_white_module_path = r""
        self.old_yellow_module_path = r""
        self.fontPath = r"C:\Users\Albert\Desktop\E_font\German.ttf"
        self.enlish_word = 'ABCDEFGHIJKLMNOPQRSTUUUVWXYZaouuu'
        self.font = ImageFont.truetype(self.fontPath, 45)
        self.font_country = ImageFont.truetype(self.fontPath, 21)
        self.new_ps1 = cv2.imdecode(np.fromfile(self.new_module_path1, dtype=np.uint8), 3)
        # self.new_ps2 = cv2.imdecode(np.fromfile(self.new_module_path2, dtype=np.uint8), 3)
        self.new_ps3 = cv2.imdecode(np.fromfile(self.new_module_path3, dtype=np.uint8), 3)
        self.mark_ps = cv2.imdecode(np.fromfile(self.mark_path, dtype=np.uint8), 3)
        self.country_number_1 = ['A', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'V', 'W', 'X', 'Y', 'Z']
        self.country_number_2 =['AB', 'AA', 'AB', 'AC', 'AE', 'AH', 'AK', 'AM', 'AN', 'Ao', 'AP', 'AS', 'AT', 'AU', 'AW', 'AZ', 'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BH', 'BI', 'BK', 'BL', 'BM', 'BN', 'BO', 'Bo', 'BP', 'BS', 'BT', 'BW', 'BZ', 'CA', 'CB', 'CE', 'CO', 'CR', 'CW', 'DA', 'DD', 'DE', 'DH', 'DI', 'DL', 'DM', 'DN', 'DO', 'DU', 'DW', 'DZ', 'EA', 'EB', 'ED', 'EE', 'EF', 'EG', 'EH', 'EI', 'EL', 'EM', 'EN', 'ER', 'ES', 'EU', 'EW', 'FB', 'FD', 'FF', 'FG', 'FI', 'FL', 'FN', 'FO', 'FR', 'FS', 'FT', 'Fu', 'FW', 'FZ', 'GA', 'GC', 'GD', 'GE', 'GF', 'GG', 'GI', 'GK', 'GL', 'GM', 'GN', 'Go', 'GP', 'GR', 'GS', 'GT', 'Gu', 'GV', 'GW', 'GZ', 'HA', 'HB', 'HC', 'HD', 'HE', 'HF', 'HG', 'HH', 'HI', 'HK', 'HL', 'HM', 'HN', 'HO', 'HP', 'HR', 'HS', 'HU', 'HV', 'HX', 'HY', 'HZ', 'IK', 'IL', 'IN', 'IZ', 'JE', 'JL', 'KA', 'KB', 'KC', 'KE', 'KF', 'KG', 'KH', 'KI', 'KK', 'KL', 'KM', 'KN', 'KO', 'KR', 'KS', 'KT', 'KU', 'KW', 'KY', 'LA', 'LB', 'LC', 'LD', 'LF', 'LG', 'LH', 'LI', 'LL', 'LM', 'LN', 'Lo', 'LP', 'LR', 'LU', 'MA', 'MB', 'MC', 'MD', 'ME', 'MG', 'MH', 'MI', 'MK', 'ML', 'MM', 'MN', 'MO', 'MQ', 'MR', 'MS', 'Mu', 'MW', 'MY', 'MZ', 'NB', 'ND', 'NE', 'NF', 'NH', 'NI', 'NK', 'NL', 'NM', 'No', 'NP', 'NR', 'NT', 'NU', 'NW', 'NY', 'NZ', 'OA', 'OB', 'OC', 'OD', 'OE', 'OF', 'OG', 'OH', 'OK', 'OL', 'OP', 'OS', 'OZ', 'PA', 'PB', 'PE', 'PF', 'PI', 'PL', 'PM', 'PN', 'PR', 'PS', 'PW', 'PZ', 'RA', 'RC', 'RD', 'RE', 'RG', 'RH', 'RI', 'RL', 'RM', 'RN', 'RO', 'RP', 'RS', 'RT', 'RU', 'RV', 'RW', 'RZ', 'SB', 'SC', 'SE', 'SG', 'SH', 'SI', 'SK', 'SL', 'SM', 'SN', 'SO', 'SP', 'SR', 'ST', 'SU', 'SW', 'SY', 'SZ', 'TE', 'TF', 'TG', 'TO', 'TP', 'TR', 'TS', 'TT', 'Tu', 'UE', 'UH', 'UL', 'UM', 'UN', 'uB', 'VB', 'VG', 'VK', 'VR', 'VS', 'WA', 'WB', 'WE', 'WF', 'WG', 'WI', 'WK', 'WL', 'WM', 'WN', 'WO', 'WR', 'WS', 'WT', 'Wu', 'WW', 'WZ', 'ZE', 'ZI', 'ZP', 'ZR', 'ZW', 'ZZ']
        self.country_number_3 = ['ABG', 'ABI', 'AIB', 'AIC', 'ALF', 'ALZ', 'ANA', 'ANG', 'ANK', 'APD', 'ARN', 'ART', 'ASL', 'ASZ', 'AUR', 'AZE', 'BAD', 'BAR', 'BBG', 'BBL', 'BCH', 'BED', 'BER', 'BGD', 'BGL', 'BID', 'BIN', 'BIR', 'BIT', 'BIW', 'BKS', 'BLB', 'BLK', 'BNA', 'BOG', 'BOH', 'BOR', 'BOT', 'BRA', 'BRB', 'BRG', 'BRK', 'BRL', 'BRV', 'BSB', 'BSK', 'BTF', 'BuD', 'BUL', 'BuR', 'BuS', 'BuZ', 'BWL', 'BYL', 'CAS', 'CHA', 'CLP', 'CLZ', 'COC', 'COE', 'CUX', 'DAH', 'DAN', 'DAU', 'DBR', 'DEG', 'DEL', 'DGF', 'DIL', 'DIN', 'DIZ', 'DKB', 'DLG', 'DON', 'DUD', 'DuW', 'EBE', 'EBN', 'EBS', 'ECK', 'EIC', 'EIL', 'EIN', 'EIS', 'EMD', 'EMS', 'ERB', 'ERH', 'ERK', 'ERZ', 'ESB', 'ESW', 'FDB', 'FDS', 'FEU', 'FFB', 'FKB', 'FLo', 'FOR', 'FRG', 'FRI', 'FRW', 'FTL', 'FuS', 'GAN', 'GAP', 'GDB', 'GEL', 'GEO', 'GER', 'GHA', 'GHC', 'GLA', 'GMN', 'GNT', 'GOA', 'GOH', 'GRA', 'GRH', 'GRI', 'GRM', 'GRZ', 'GTH', 'GUB', 'GUN', 'GVM', 'HAB', 'HAL', 'HAM', 'HAS', 'HBN', 'HBS', 'HCH', 'HDH', 'HDL', 'HEB', 'HEF', 'HEI', 'HEL', 'HER', 'HET', 'HGN', 'HGW', 'HHM', 'HIG', 'HIP', 'HMu', 'HOG', 'HOH', 'HOL', 'HOM', 'HOR', 'HoS', 'HOT', 'HRO', 'HSK', 'HST', 'HVL', 'HWI', 'IGB', 'ILL', 'JuL', 'KEH', 'KEL', 'KEM', 'KIB', 'KLE', 'KLZ', 'KoN', 'KoT', 'KoZ', 'KRU', 'KuN', 'KUS', 'KYF', 'LAN', 'LAU', 'LBS', 'LBZ', 'LDK', 'LDS', 'LEO', 'LER', 'LEV', 'LIB', 'LIF', 'LIP', 'LoB', 'LOS', 'LRO', 'LSA', 'LSN', 'LSZ', 'LuN', 'LUP', 'LWL', 'MAB', 'MAI', 'MAK', 'MAL', 'MED', 'MEG', 'MEI', 'MEK', 'MEL', 'MER', 'MET', 'MGH', 'MGN', 'MHL', 'MIL', 'MKK', 'MOD', 'MOL', 'MON', 'MOS', 'MSE', 'MSH', 'MSP', 'MST', 'MTK', 'MTL', 'MuB', 'MuR', 'MVL', 'MYK', 'MZG', 'NAB', 'NAI', 'NAU', 'NDH', 'NEA', 'NEB', 'NEC', 'NEN', 'NES', 'NEW', 'NMB', 'NMS', 'NOH', 'NOL', 'NOM', 'NOR', 'NRW', 'NVP', 'NWM', 'OAL', 'OBB', 'OBG', 'OCH', 'OHA', 'oHR', 'OHV', 'OHZ', 'OPR', 'OSL', 'OVI', 'OVL', 'OVP', 'PAF', 'PAN', 'PAR', 'PCH', 'PEG', 'PIR', 'PLo', 'PRu', 'QFT', 'QLB', 'RDG', 'REG', 'REH', 'REI', 'RID', 'RIE', 'ROD', 'ROF', 'ROK', 'ROL', 'ROS', 'ROT', 'ROW', 'RPL', 'RSL', 'RuD', 'RuG', 'SAB', 'SAD', 'SAL', 'SAN', 'SAW', 'SBG', 'SBK', 'SCZ', 'SDH', 'SDL', 'SDT', 'SEB', 'SEE', 'SEF', 'SEL', 'SFB', 'SFT', 'SGH', 'SHA', 'SHG', 'SHK', 'SHL', 'SIG', 'SIH', 'SIM', 'SLE', 'SLF', 'SLG', 'SLK', 'SLN', 'SLS', 'SLu', 'SLZ', 'SMu', 'SOB', 'SOG', 'SOK', 'SoM', 'SON', 'SPB', 'SPN', 'SRB', 'SRO', 'STA', 'STB', 'STD', 'STE', 'STL', 'SUL', 'SuW', 'SWA', 'SZB', 'TBB', 'TDO', 'TET', 'THL', 'THW', 'TIR', 'ToL', 'TUT', 'UEM', 'UFF', 'USI', 'VAI', 'VEC', 'VER', 'VIB', 'VIE', 'VIT', 'VOH', 'WAF', 'WAK', 'WAN', 'WAT', 'WBS', 'WDA', 'WEL', 'WEN', 'WER', 'WES', 'WHV', 'WIL', 'WIN', 'WIS', 'WIT', 'WIV', 'WIZ', 'WLG', 'WMS', 'WND', 'WOB', 'WOH', 'WOL', 'WOR', 'WOS', 'WRN', 'WSF', 'WST', 'WSW', 'WTL', 'WTM', 'WUG', 'WuM', 'WUN', 'WUR', 'WZL', 'ZEL', 'ZIG']



    def get_word2(self, num_word1, num_word23):
        word_2 = ''
        for i in range(num_word23[0]):
            word_tmp = ''.join(random.choices(self.enlish_word))
            word_2 += word_tmp
        word_3 = ''.join(random.choices(string.digits, k=num_word23[1]))
        word_2_name = word_2 + word_3
        if num_word1 + num_word23[0] + num_word23[1] == 8:
            word_2 = word_2 + '.' + word_3
        else:
            word_2 = word_2 + '-' + word_3

        return word_2, word_2_name
    def calculate_word_distence(self, num_1, num_word23):
        num_2 = num_word23[0]
        num_3 = num_word23[1]
        all_d = 229
        front_mark_d = 25
        english_word_d = 25
        mark_width = 21
        mark_front_d = 2
        center_word_d = (num_1 + num_2 + num_3) * english_word_d + mark_width + mark_front_d + mark_front_d
        word_1_d = front_mark_d + int((all_d - center_word_d) / 2) -1
        word_2_d = word_1_d + num_1 * english_word_d + mark_front_d + mark_width + mark_front_d
        mark_d = word_1_d + num_1 * english_word_d + mark_front_d
        return word_1_d, word_2_d, mark_d

    def check_ban_word(self, word):
        ban_word = ['HJ', 'KZ', 'NS', 'SA', 'SS', '188', '888', '1888', '8818', '8888']
        for word_tmp in ban_word:
            if word_tmp in word:
                return False
        return True


    def generator_all(self):

        num_english_word = 3
        choice_list_1 = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [2, 3], [2, 4]]
        choice_list_2 = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [2, 3], [2, 4]]
        choice_list_3 = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [2, 3]]

        word_color = (0, 0, 0)
        img_copy = self.new_ps1.copy()
        random_num = random.randint(1,3)
        # random_num = 2
        if random_num == 1:
            word_1 = self.country_number_1[random.randint(0, len(self.country_number_1)-1)]
            num_word23 = choice_list_1[random.randint(0, len(choice_list_1) - 1)]
        elif random_num == 2:
            word_1 = self.country_number_2[random.randint(0, len(self.country_number_2) - 1)]
            num_word23 = choice_list_2[random.randint(0, len(choice_list_2) - 1)]
        elif random_num == 3:
            word_1 = self.country_number_3[random.randint(0, len(self.country_number_3) - 1)]
            num_word23 = choice_list_3[random.randint(0, len(choice_list_3) - 1)]
        print(word_1)
        # num_word23 = [2,4]
        word_2, word_2_name = self.get_word2(random_num, num_word23)
        while not self.check_ban_word(word_2_name):
            word_2, word_2_name = self.get_word2(random_num, num_word23)

        # word_2 = 'MM-4444'
        word_1_d, word_2_d, mark_d = self.calculate_word_distence(random_num, num_word23)
        img_copy[8:49, mark_d:mark_d + 21, 0:3] = self.mark_ps
        imgPil = Image.fromarray(img_copy)
        draw = ImageDraw.Draw(imgPil)
        draw.text((word_1_d, 1), word_1, font=self.font, fill=word_color)
        draw.text((word_2_d, 1), word_2, font=self.font, fill=word_color)

        img_copy = np.array(imgPil)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        # img_copy = cv2.imdecode(np.fromfile(r"C:\Users\Albert\Desktop\123.jpg", dtype=np.uint8), 3)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image_Process().data_augmentation_keras(img_copy, 150, [0.01, 1.8])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        word_all = word_1 + word_2_name
        filename = "{}_GAN".format(word_all)
        # filename = r"{}\{}".format(self.out_dir, filename)
        # cv2.imwrite(filename, img_copy)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename


class California_LP_Module():
    def __init__(self):
        self.out_dir = r"D:\data\data_4GAN\California\GAN_result"
        self.new_module_path1 = r"D:\data\data_4GAN\California\ca_new_ps1.png"
        self.new_module_path2 = r"D:\data\data_4GAN\California\ca_new_ps2.png"
        self.dis_module_path1 = r"D:\data\data_4GAN\California\ca_dis_ps1.jpg"
        self.dis_module_path2 = r"D:\data\data_4GAN\California\ca_dis_ps2.jpg"
        self.black_module_path1 = r"D:\data\data_4GAN\California\ca_black_ps1.jpg"
        self.black_module_path2 = r"D:\data\data_4GAN\California\ca_black_ps2.jpg"
        # self.new_module_path2 = r"D:\data\data_4GAN\California\new_ps2.jpg"
        self.old_white_module_path = r""
        self.old_yellow_module_path = r""
        self.fontPath = r"C:\Users\Albert\Desktop\CA_font\dealerplate california1.ttf"
        self.EN_word_for_train = 'AAABCCDEEEEEFFFFFFGGHIIIJKLLMMNOOOOPQQQQQQQQRSTTTTUVWWXY'
        self.NUM_word_for_train = '0000011112344456789'
        self.font = ImageFont.truetype(self.fontPath, 57)
        self.new_ps1 = cv2.imdecode(np.fromfile(self.new_module_path1, dtype=np.uint8), 3)
        self.new_ps2 = cv2.imdecode(np.fromfile(self.new_module_path2, dtype=np.uint8), 3)
        self.dis_ps1 = cv2.imdecode(np.fromfile(self.dis_module_path1, dtype=np.uint8), 3)
        self.dis_ps2 = cv2.imdecode(np.fromfile(self.dis_module_path2, dtype=np.uint8), 3)
        self.black_ps1 = cv2.imdecode(np.fromfile(self.black_module_path1, dtype=np.uint8), 3)
        self.black_ps2 = cv2.imdecode(np.fromfile(self.black_module_path2, dtype=np.uint8), 3)
        # self.new_ps2 = cv2.imdecode(np.fromfile(self.new_module_path2, dtype=np.uint8), 3)
    def generator_new1(self):
        num_english_word = 3
        img_copy = self.new_ps1.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''.join(random.choices(self.NUM_word_for_train))
        word_2 = ''
        for index_word in range(num_english_word):
            word_tmp = ''.join(random.choices(self.EN_word_for_train))
            word_2 += word_tmp
        word_3 = ''
        for index_word in range(3):
            word_tmp = ''.join(random.choices(self.NUM_word_for_train))
            word_3 += word_tmp


        draw = ImageDraw.Draw(imgPil)
        word_color = (97, 18, 31)
        label = word_1 + word_2 + word_3
        word = '-'.join(label)
        draw.text((12, 37), word, font=self.font, fill=word_color)
        # (250, 232, 215)
        # draw.text((112, 11), word_2, font=self.font, fill=word_color)
        # draw.text((153, 7), word_3, font=self.font, fill=word_color)
        # word_contry = self.get_country_number()
        # word_color = (255, 255, 255)
        # draw.text((273, 35), word_contry, font=self.font_country, fill=word_color)
        img_copy = np.array(imgPil)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image_Process().data_augmentation_keras(img_copy, 100, [0.1, 1.8])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename
    def generator_new2(self):
        num_english_word = 3
        img_copy = self.new_ps2.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''.join(random.choices(self.NUM_word_for_train))
        word_2 = ''
        for index_word in range(num_english_word):
            word_tmp = ''.join(random.choices(self.EN_word_for_train))
            word_2 += word_tmp
        word_3 = ''
        for index_word in range(3):
            word_tmp = ''.join(random.choices(self.NUM_word_for_train))
            word_3 += word_tmp


        draw = ImageDraw.Draw(imgPil)
        word_color = (97, 18, 31)
        label = word_1 + word_2 + word_3
        word = '-'.join(label)

        draw.text((6, 22), word, font=self.font, fill=word_color)
        # (250, 232, 215)
        # draw.text((112, 11), word_2, font=self.font, fill=word_color)
        # draw.text((153, 7), word_3, font=self.font, fill=word_color)
        # word_contry = self.get_country_number()
        # word_color = (255, 255, 255)
        # draw.text((273, 35), word_contry, font=self.font_country, fill=word_color)
        img_copy = np.array(imgPil)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image_Process().data_augmentation_keras(img_copy, 100, [0.1, 1.8])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename
    def generator_dis1(self):
        num_english_word = 2
        img_copy = self.dis_ps1.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''
        for index_word in range(3):
            word_tmp = ''.join(random.choices(self.NUM_word_for_train))
            word_1 += word_tmp
        word_2 = ''
        for index_word in range(num_english_word):
            word_tmp = ''.join(random.choices(self.EN_word_for_train))
            word_2 += word_tmp



        draw = ImageDraw.Draw(imgPil)
        word_color = (107, 62, 48)
        label = word_1 + word_2
        # label = "822BR"
        word = '-'.join(label)

        draw.text((62, 41), word, font=self.font, fill=word_color)
        # (250, 232, 215)
        # draw.text((112, 11), word_2, font=self.font, fill=word_color)
        # draw.text((153, 7), word_3, font=self.font, fill=word_color)
        # word_contry = self.get_country_number()
        # word_color = (255, 255, 255)
        # draw.text((273, 35), word_contry, font=self.font_country, fill=word_color)
        img_copy = np.array(imgPil)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image_Process().data_augmentation_keras(img_copy, 100, [0.1, 1.5])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_dis_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename
    def generator_dis2(self):
        num_english_word = 2
        img_copy = self.dis_ps2.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''
        for index_word in range(4):
            word_tmp = ''.join(random.choices(self.NUM_word_for_train))
            word_1 += word_tmp
        word_2 = ''.join(random.choices(self.EN_word_for_train))

        draw = ImageDraw.Draw(imgPil)
        word_color = (58, 45, 36)
        label = word_1 + word_2
        word = '-'.join(label)

        draw.text((47, 38), word, font=self.font, fill=word_color)
        # (250, 232, 215)
        # draw.text((112, 11), word_2, font=self.font, fill=word_color)
        # draw.text((153, 7), word_3, font=self.font, fill=word_color)
        # word_contry = self.get_country_number()
        # word_color = (255, 255, 255)
        # draw.text((273, 35), word_contry, font=self.font_country, fill=word_color)
        img_copy = np.array(imgPil)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image_Process().data_augmentation_keras(img_copy, 100, [0.1, 1.5])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_dis_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename
    def generator_black1(self):
        first_english_word = 'BCDEFGHIJKLMNOPQRSTUVW'
        last_english_word = 'ABCDEFG'
        font = ImageFont.truetype(self.fontPath, 53)
        num_english_word = 2
        img_copy = self.black_ps1.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''.join(random.choices(first_english_word))
        word_2 = ''.join(random.choices(string.digits, k=3))
        word_3 = ''.join(random.choices(last_english_word))
        word_4 = ''.join(random.choices(string.digits, k=1))

        draw = ImageDraw.Draw(imgPil)
        word_color = (58, 204, 248)
        label = word_1 + word_2 + word_3 + word_4
        word = '-'.join(label)
        # word = 'Q-1-2-3-Q-1'
        draw.text((24, 36), word, font=font, fill=word_color)
        # (250, 232, 215)
        # draw.text((112, 11), word_2, font=self.font, fill=word_color)
        # draw.text((153, 7), word_3, font=self.font, fill=word_color)
        # word_contry = self.get_country_number()
        # word_color = (255, 255, 255)
        # draw.text((273, 35), word_contry, font=self.font_country, fill=word_color)
        img_copy = np.array(imgPil)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image_Process().data_augmentation_keras(img_copy, 100, [0.1, 1.5])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_dis_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename
    def generator_black2(self):
        font = ImageFont.truetype(self.fontPath, 53)

        img_copy = self.black_ps2.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''.join(random.choices(string.ascii_uppercase, k=3))
        word_2 = ''.join(random.choices(string.digits, k=3))


        draw = ImageDraw.Draw(imgPil)
        word_color = (6, 170, 213)
        label = word_1 + word_2
        word = word_1[0] + '---' + word_1[1] + '--' + word_1[2] + '----' + '-'.join(word_2)
        # word = 'S---A--M----1-2-3'
        draw.text((12, 33), word, font=font, fill=word_color)
        # (250, 232, 215)
        # draw.text((112, 11), word_2, font=self.font, fill=word_color)
        # draw.text((153, 7), word_3, font=self.font, fill=word_color)
        # word_contry = self.get_country_number()
        # word_color = (255, 255, 255)
        # draw.text((273, 35), word_contry, font=self.font_country, fill=word_color)
        img_copy = np.array(imgPil)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image_Process().data_augmentation_keras(img_copy, 100, [0.3, 1.5])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_dis_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename

    def generator_all(self):
        switch = random.randint(0,7)
        # switch = 9
        switch = 1
        if switch ==0:
            switch2 = random.randint(0, 2)
            if switch2 == 2:
                A, B = self.generator_dis2()
            else:
                A, B = self.generator_dis1()
        elif switch ==1:
            switch2 = random.randint(0, 2)
            if switch2 == 2:
                A, B = self.generator_black2()
            else:
                A, B = self.generator_black1()
        else:
            A, B = self.generator_new1()
        return A, B

class Vietnam_LP_Module():
    def __init__(self):
        self.out_dir = r"D:\data\data_4GAN\Vietnam\GAN_result"
        self.module_path1 = r"D:\data\data_4GAN\Vietnam\VN_332_ps.jpg"
        self.module_path2 = r"D:\data\data_4GAN\Vietnam\VN_332_white_ps.jpg"
        self.module_path3 = r"D:\data\data_4GAN\Vietnam\VN_232_ps.jpg"
        self.module_path4 = r"D:\data\data_4GAN\Vietnam\VN_222_ps.jpg"
        self.module_path5 = r"D:\data\data_4GAN\Vietnam\VN_222_white_ps.jpg"
        self.module_path6 = r"D:\data\data_4GAN\Vietnam\VN34.jpg"
        self.module_path7 = r"D:\data\data_4GAN\Vietnam\VN_222EN_ps.jpg"
        self.module_path8 = r"D:\data\data_4GAN\Vietnam\VN_UN_ps.jpg"
        self.module_path9 = r"D:\data\data_4GAN\Vietnam\VN_44EN_ps.jpg"
        self.old_white_module_path = r""
        self.old_yellow_module_path = r""
        self.fontPath = r"C:\Users\Albert\Desktop\font\vietnam\Vietnam.ttf"
        self.first_num_word = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
                               '40', '41', '43', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72',
                               '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '88', '89', '90', '92', '93', '94', '95', '97', '98', '99']
        self.EN_word_for_train = 'ABCDEFGHKLMNOPQRSTUVXYZ'
        self.NUM_word_for_train = '0123456789'
        self.EN_44 = ['DA', 'HC', 'KT', 'LA', 'LD', 'MA', 'MD', 'MK', 'SA', 'XA', 'TD']
        self.EN_222 = ['AT', 'BB', 'BC', 'BH', 'BT', 'BP', 'HB', 'HH', 'KA', 'KB', 'KC', 'KD', 'KV', 'KP', 'KK', 'PP', 'QH', 'QK', 'QP', 'TC', 'TH', 'TK', 'TT', 'TM', 'VT']
        self.font = ImageFont.truetype(self.fontPath, 43)
        self.new_ps1 = cv2.imdecode(np.fromfile(self.module_path1, dtype=np.uint8), 3)
        self.new_ps2 = cv2.imdecode(np.fromfile(self.module_path2, dtype=np.uint8), 3)
        self.new_ps3 = cv2.imdecode(np.fromfile(self.module_path3, dtype=np.uint8), 3)
        self.new_ps4 = cv2.imdecode(np.fromfile(self.module_path4, dtype=np.uint8), 3)
        self.new_ps5 = cv2.imdecode(np.fromfile(self.module_path5, dtype=np.uint8), 3)
        self.new_ps6 = cv2.imdecode(np.fromfile(self.module_path6, dtype=np.uint8), 3)
        self.new_ps7 = cv2.imdecode(np.fromfile(self.module_path7, dtype=np.uint8), 3)
        self.new_ps8 = cv2.imdecode(np.fromfile(self.module_path8, dtype=np.uint8), 3)
        self.new_ps9 = cv2.imdecode(np.fromfile(self.module_path9, dtype=np.uint8), 3)

        # self.new_ps2 = cv2.imdecode(np.fromfile(self.new_module_path2, dtype=np.uint8), 3)
    def generator_332(self):
        num_english_word = 3
        img_copy = self.new_ps1.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''.join(random.choices(self.first_num_word))
        # word_1 += ''.join(random.choices(self.EN_word_for_train, k=1))
        word_1 += 'LD'
        word_2 = ''.join(random.choices(self.NUM_word_for_train, k=3))
        word_3 = ''.join(random.choices(self.NUM_word_for_train, k=2))
        label = word_1 + word_2 + word_3
        font = ImageFont.truetype(self.fontPath, 47)
        draw = ImageDraw.Draw(imgPil)
        word_color = (48, 48, 48)
        # word_color = (255, 0, 0)
        # word_1 = '35H'
        # word_2 = '020'
        # word_3 = '27'
        draw.text((4, 4), word_1, font=font, fill=word_color)
        draw.text((79, 4), word_2, font=font, fill=word_color)
        draw.text((150, 4), word_3, font=font, fill=word_color)
        # (250, 232, 215)
        # draw.text((112, 11), word_2, font=self.font, fill=word_color)
        # draw.text((153, 7), word_3, font=self.font, fill=word_color)
        # word_contry = self.get_country_number()
        # word_color = (255, 255, 255)
        # draw.text((273, 35), word_contry, font=self.font_country, fill=word_color)
        img_copy = np.array(imgPil)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image_Process().data_augmentation_keras(img_copy, 80, [0.1, 1.8])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename
    def generator_332_white(self):
        num_english_word = 3
        img_copy = self.new_ps2.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''.join(random.choices(self.first_num_word))
        word_1 += ''.join(random.choices(self.EN_word_for_train, k=1))
        word_2 = ''.join(random.choices(self.NUM_word_for_train, k=3))
        word_3 = ''.join(random.choices(self.NUM_word_for_train, k=2))
        label = word_1 + word_2 + word_3
        font = ImageFont.truetype(self.fontPath, 47)
        draw = ImageDraw.Draw(imgPil)
        word_color = (141, 141, 141)
        draw.text((4, 4), word_1, font=font, fill=word_color)
        draw.text((79, 4), word_2, font=font, fill=word_color)
        draw.text((150, 4), word_3, font=font, fill=word_color)
        img_copy = np.array(imgPil)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        # img_copy = Image_Process().data_augmentation_keras(img_copy, 20, [0.1, 1.1])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename
    def generator_232(self):
        num_english_word = 3
        img_copy = self.new_ps3.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''.join(random.choices(string.digits, k=1))
        word_1 += ''.join(random.choices(self.EN_word_for_train, k=1))
        word_2 = ''.join(random.choices(self.NUM_word_for_train, k=3))
        word_3 = ''.join(random.choices(self.NUM_word_for_train, k=2))
        label = word_1 + word_2 + word_3
        font = ImageFont.truetype(self.fontPath, 49)
        draw = ImageDraw.Draw(imgPil)
        word_color = (21, 21, 21)
        draw.text((4, 3), word_1, font=font, fill=word_color)
        draw.text((66, 3), word_2, font=font, fill=word_color)
        draw.text((150, 3), word_3, font=font, fill=word_color)
        img_copy = np.array(imgPil)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image_Process().data_augmentation_keras(img_copy, 60, [0.1, 1.3])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename
    def generator_222(self):
        num_english_word = 3
        img_copy = self.new_ps4.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''.join(random.choices(string.digits, k=1))
        word_1 += ''.join(random.choices(self.EN_word_for_train, k=1))
        word_2 = ''.join(random.choices(self.NUM_word_for_train, k=2))
        word_3 = ''.join(random.choices(self.NUM_word_for_train, k=2))
        label = word_1 + word_2 + word_3
        font = ImageFont.truetype(self.fontPath, 47)
        draw = ImageDraw.Draw(imgPil)
        word_color = (71, 71, 71)
        draw.text((9, 4), word_1, font=font, fill=word_color)
        draw.text((77, 4), word_2, font=font, fill=word_color)
        draw.text((144, 4), word_3, font=font, fill=word_color)
        img_copy = np.array(imgPil)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image_Process().data_augmentation_keras(img_copy, 80, [0.1, 1.8])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename
    def generator_222_white(self):
        num_english_word = 3
        img_copy = self.new_ps5.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''.join(random.choices(string.digits, k=1))
        word_1 += ''.join(random.choices(self.EN_word_for_train, k=1))
        word_2 = ''.join(random.choices(self.NUM_word_for_train, k=2))
        word_3 = ''.join(random.choices(self.NUM_word_for_train, k=2))
        label = word_1 + word_2 + word_3
        font = ImageFont.truetype(self.fontPath, 47)
        draw = ImageDraw.Draw(imgPil)
        word_color = (244, 244, 244)
        draw.text((9, 4), word_1, font=font, fill=word_color)
        draw.text((77, 4), word_2, font=font, fill=word_color)
        draw.text((144, 4), word_3, font=font, fill=word_color)
        img_copy = np.array(imgPil)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        # img_copy = Image_Process().data_augmentation_keras(img_copy, 80, [0.5, 1.5])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename
    def generator_34(self):
        num_english_word = 3
        img_copy = self.new_ps6.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''.join(random.choices(self.first_num_word))
        word_1 += ''.join(random.choices(self.EN_word_for_train, k=1))
        word_2 = ''.join(random.choices(self.NUM_word_for_train, k=4))
        label = word_1 + word_2
        font = ImageFont.truetype(self.fontPath, 46)
        draw = ImageDraw.Draw(imgPil)
        word_color = (82, 82, 82)
        draw.text((16, 6), word_1, font=font, fill=word_color)
        draw.text((93, 6), word_2, font=font, fill=word_color)
        # draw.text((150, 4), word_3, font=font, fill=word_color)
        # (250, 232, 215)
        # draw.text((112, 11), word_2, font=self.font, fill=word_color)
        # draw.text((153, 7), word_3, font=self.font, fill=word_color)
        # word_contry = self.get_country_number()
        # word_color = (255, 255, 255)
        # draw.text((273, 35), word_contry, font=self.font_country, fill=word_color)
        img_copy = np.array(imgPil)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image_Process().data_augmentation_keras(img_copy, 80, [0.1, 1.8])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename
    def generator_222EN(self):
        #軍用車
        num_english_word = 3
        img_copy = self.new_ps7.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''.join(random.choices(self.EN_222, k=1))
        word_2 = ''.join(random.choices(self.NUM_word_for_train, k=2))
        word_3 = ''.join(random.choices(self.NUM_word_for_train, k=2))
        label = word_1 + word_2 + word_3
        font = ImageFont.truetype(self.fontPath, 47)
        draw = ImageDraw.Draw(imgPil)
        word_color = (62, 62, 62)
        # word_color = (255, 0, 0)
        draw.text((9, 4), word_1, font=font, fill=word_color)
        draw.text((77, 4), word_2, font=font, fill=word_color)
        draw.text((144, 4), word_3, font=font, fill=word_color)
        img_copy = np.array(imgPil)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image_Process().data_augmentation_keras(img_copy, 80, [0.1, 1.8])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename
    def generator_UN(self):
        num_english_word = 3
        img_copy = self.new_ps8.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''.join(random.choices(self.NUM_word_for_train, k=4))
        label = 'UN' + word_1
        font = ImageFont.truetype(self.fontPath, 45)
        draw = ImageDraw.Draw(imgPil)
        word_color = (36, 36, 36)
        # word_color = (255, 0, 0)
        draw.text((61, 29), word_1, font=font, fill=word_color)
        # draw.text((144, 4), word_3, font=font, fill=word_color)
        img_copy = np.array(imgPil)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image_Process().data_augmentation_keras(img_copy, 80, [0.1, 1.8])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename
    def generator_44EN(self):
        #商用車
        num_english_word = 3
        img_copy = self.new_ps9.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''.join(random.choices(self.first_num_word, k=1))
        word_1 += ''.join(random.choices(self.EN_44, k=1))
        word_2 = ''.join(random.choices(self.NUM_word_for_train, k=4))
        label = word_1 + word_2
        font = ImageFont.truetype(self.fontPath, 50)
        draw = ImageDraw.Draw(imgPil)
        word_color = (67, 67, 67)
        # word_color = (255, 0, 0)
        # word_1 = '29LD'
        # word_2 = '2926'
        draw.text((6, 3), word_1, font=font, fill=word_color)
        draw.text((107, 3), word_2, font=font, fill=word_color)
        img_copy = np.array(imgPil)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image_Process().data_augmentation_keras(img_copy, 60, [0.3, 1.7])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename
    def generator_all(self):
        switch = random.randint(0,1)
        # switch = 9
        switch = 1
        if switch ==0:
                A, B = self.generator_332()
        else:
            switch2 = random.randint(6, 7)
            switch2=7
            if switch2 == 0:
                A, B = self.generator_332_white()
            elif switch2 ==1:
                A, B = self.generator_232()
            elif switch2 ==2:
                A, B = self.generator_222()
            elif switch2 ==3:
                A, B = self.generator_222_white()
            elif switch2 ==4:
                A, B = self.generator_34()
            elif switch2 ==5:
                A, B = self.generator_222EN()
            elif switch2 ==6:
                A, B = self.generator_UN()
            elif switch2 ==7:
                A, B = self.generator_44EN()
        return A, B

class Taiwan_LP_Module():
    def __init__(self):
        self.out_dir = r"D:\data\data_4GAN\Taiwan\GAN_result"
        self.fontPath = r"C:\Users\Albert\Desktop\T_font\Taiwan1.ttf"
        self.module_path1 = r"D:\data\data_4GAN\Taiwan\military_ps1.jpg"
        self.new_ps1 = cv2.imdecode(np.fromfile(self.module_path1, dtype=np.uint8), 3)
        self.EN_word_for_train = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.NUM_word_for_train = '0123456789'
        self.font = ImageFont.truetype(self.fontPath, 77)
    def generator_military(self):
        img_copy = self.new_ps1.copy()
        imgPil = Image.fromarray(img_copy)
        word_1 = ''.join(random.choices(self.EN_word_for_train, k=1))
        word_2 = '10'
        word_2 += ''.join(random.choices(self.NUM_word_for_train, k=3))
        label = word_1 + word_2
        font = ImageFont.truetype(self.fontPath, 4)
        draw = ImageDraw.Draw(imgPil)
        word_color = (232, 242, 234)
        # word_color = (255, 0, 0)
        # word_1 = 'J'
        # word_2 = '10071'
        # word_3 = '27'
        draw.text((30, 0), word_1, font=self.font, fill=word_color)
        draw.text((67, 0), word_2, font=self.font, fill=word_color)
        # (250, 232, 215)
        # draw.text((112, 11), word_2, font=self.font, fill=word_color)
        # draw.text((153, 7), word_3, font=self.font, fill=word_color)
        # word_contry = self.get_country_number()
        # word_color = (255, 255, 255)
        # draw.text((273, 35), word_contry, font=self.font_country, fill=word_color)
        img_copy = np.array(imgPil)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image_Process().data_augmentation_keras(img_copy, 80, [0.1, 1.3])
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img_copy)
        # cv2.waitKey(0)
        filename = "{}_GAN".format(label)
        # filename = r"{}\{}{}{}_GAN.jpg".format(self.out_dir, word_1, word_2, word_3)
        return img_copy, filename

    def generator_all(self):
        img, filename = self.generator_military()
        return img, filename

if __name__ == '__main__':
    pass
    # UK_LP_Module().generator_new2()
    # German_LP_Module().generator_all()
    # Vietnam_LP_Module().generator_44EN()
    Taiwan_LP_Module().generator_military()