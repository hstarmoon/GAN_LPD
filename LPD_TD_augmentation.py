import cv2
import numpy as np
import glob, shutil, os, random
from image_process import Random_Process, Image_Process


def augmentation_img():
    img_path = r"D:\data\USA_CA\USA_CA_source_new\train_TD\train_TD_e"
    file_foder_list = os.listdir(img_path)
    for forder in file_foder_list:
        img_list = glob.glob(r"{}\{}\*.jpg".format(img_path, forder))
        for index, img_name in enumerate(img_list):
            img = cv2.imread(img_name)
            basename = os.path.basename(img_name)[:-4]
            random_choice = random.randint(0,7)
            if random_choice == 7:
                random_choice2 = random.randint(0,1)
                if random_choice2 == 0:
                    add_rate_shadow = random.randint(20, 70) * 0.01
                    mode = random.randint(0, 4)
                    img_result = Random_Process().random_shadow_img(img, add_rate_shadow, 0, mode)
                else:
                    bios = 0
                    radius = random.randint(80, 250)
                    add_rate = random.randint(15, 50) * 0.01
                    img_result = Random_Process().random_sun_flare_img(img, add_rate, bios,
                                                                no_of_flare_circles=0, src_radius=radius)

                cv2.imwrite(r"{}\{}\{}_sh.jpg".format(img_path, forder, basename), img_result)
                shutil.copyfile(r"{}\{}\{}.txt".format(img_path, forder, basename), r"{}\{}\{}_sh.txt".format(img_path, forder, basename))



            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.namedWindow("test", 0)
            # cv2.imshow("test", img)
            # img_dark = darken(img)
            # cv2.namedWindow("dark", 0)
            # cv2.imshow("dark", img_dark)
            # cv2.waitKey(0)


            # cv2.imwrite(r"{}\sun_{}".format(out_path, basename), img_sun)

if __name__ == '__main__':
    augmentation_img()