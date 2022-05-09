#coding=utf-8
import cv2
import numpy as np
import glob, shutil, os, random
from PIL import Image

# from Automold import darken
# from image_process import Random_Process, Image_Process, string
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
def test():
    file = r"C:\Users\Albert\Desktop\T_font\A"
    kernel = np.ones((7,7),np.uint8)

    img = cv2.imread("{}.jpg".format(file), 0)
    img_w=img.shape[1]
    img_h=img.shape[0]
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("gray", 0)
    # cv2.resizeWindow("gray", img_w*5, img_h*5)
    cv2.imshow("gray", img)


    _, img_gray = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    cv2.namedWindow("th", 0)
    # cv2.resizeWindow("th", img_w*5, img_h*5)
    cv2.imshow("th", img_gray)
    cv2.waitKey(0)
    cv2.imwrite(r"{}_new.png".format(file), img_gray)
    # img_open = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
    # cv2.namedWindow("open", 0)
    # cv2.imshow("open", img_open)
    # cv2.waitKey(0)
    # img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel)
    # cv2.namedWindow("close", 0)
    # cv2.imshow("close", img_close)
    # cv2.imwrite(r"C:\Users\Albert\Desktop\out.jpg", img_close)
    # cv2.waitKey(0)
    a= 0
def get_country_num():
    file_path = r"D:\data\data_4GAN\EVS\German\local_number.txt"
    word_1 = []
    word_2 = []
    word_3 = []
    with open(file_path , 'r', errors='ignore') as file:
        lines = file.readlines()
        for line in lines:
            word = line.split(" ")[0]
            if len(word) == 1:
                word_1.append(word)
            elif len(word) == 2:
                word_2.append(word)
            elif len(word) == 3:
                word_3.append(word)
        print(word_1)
        print("--------------------------------")
        print(word_2)
        print("--------------------------------")
        print(word_3)
        print("--------------------------------")
        for index2 in range(len(word_2)):
            for index3 in range(len(word_2)):
                if index2 != index3:
                    if word_2[index2] == word_2[index3]:
                        print(index2)
        for index2 in range(len(word_3)):
            for index3 in range(len(word_3)):
                if index2 != index3:
                    if word_3[index2] == word_3[index3]:
                        print(index2)


def random_choice_from_GAN_pre():
    input_path = r"D:\data\Taiwan\military\T_military_all\1"
    output_path = r"D:\data\Taiwan\military\T_military_all_train"
    num_choice = 313
    img_list = glob.glob(r"{}\*.jpg".format(input_path))
    random.shuffle(img_list)
    img_list = img_list[0:num_choice]
    num = 0
    for img_path in img_list:
        img_name = os.path.basename(img_path)
        # txt_file = img_path[:-4]+'.txt'
        # new_txt_file = r"{}\{}".format(output_path, os.path.basename(txt_file))
        new_path = r"{}\{}".format(output_path, img_name)
        file_check = os.path.isfile(new_path)
        if not file_check:
            # shutil.move(img_path, new_path)
            # shutil.move(txt_file, new_txt_file)
            shutil.copyfile(img_path, new_path)
            num += 1
            # if num >= 120000:
            #     break
    
def darken_img():
    img_path = r"D:\data\USA_CA\USA_CA_source_new\train_TD\CA0938"
    img_list = glob.glob(r"{}\*.jpg".format(img_path))
    for img_name in img_list:
        img = cv2.imread(img_name)
        basename = os.path.basename(img_name)
        bios = 0
        mode = random.randint(0, 4)
        radius = random.randint(80, 250)
        add_rate = random.randint(15, 50) * 0.01
        add_rate_shadow = random.randint(20, 70) * 0.01
        img_shadow = Random_Process().random_shadow_img(img, add_rate_shadow, 0, mode)
        img_sun = Random_Process().random_sun_flare_img(img, add_rate, bios,
                                                           no_of_flare_circles=0, src_radius=radius)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.namedWindow("test", 0)
        # cv2.imshow("test", img)
        # img_dark = darken(img)
        # cv2.namedWindow("dark", 0)
        # cv2.imshow("dark", img_dark)
        # cv2.waitKey(0)
        out_path = r"D:\data\USA_CA\USA_CA_source_new\train_TD\result"
        cv2.imwrite(r"{}\shadow_{}".format(out_path, basename), img_shadow)
        cv2.imwrite(r"{}\sun_{}".format(out_path, basename), img_sun)

def skan_folder():
    dir = r"D:\data\all_TD\all_test_TD_35\EVS_test"
    new_dir = r"D:\data\EVS\clean"
    out_dir = r"D:\data\EVS\train_test"
    img_folders = glob.glob(r"{}/*.jpg".format(dir))
    folder_list = glob.glob(os.path.join(dir, r'*'))
    for folder in folder_list:
        folder_name = os.path.basename(folder)
        shutil.move(os.path.join(new_dir, folder_name), os.path.join(out_dir, folder_name))

    aaa = 0
    # for img in img_folders:
    #     img_name = os.path.basename(img)
    #     new_img_name = img_name.upper()
    #     if "_scoo" in img_name:
    #         aaa=0
    #         os.remove(img)
        # new_folder_path = os.path.join(new_dir, folder_name)
        # createFolder(new_folder_path)
        # img_list = glob.glob(r"{}/*.jpg".format(img_folder))
        # num = 0
        # num2 = 0
        # for img_path in img_list:
        #     img = cv2.imread(img_path)
        #     (img_h, img_w) = img.shape[:2]
        #     module = np.zeros((1080, 1920, 3), dtype=np.uint8)
        #     (module_h, module_w) = module.shape[:2]
        #     module[int((module_h-img_h)/2):int((module_h-img_h)/2)+img.shape[0], int((module_w-img_w)/2):int((module_w-img_w)/2)+img.shape[1]] = img
            # cv2.namedWindow("test", 0)
            # cv2.imshow("test", module)
            # cv2.waitKey(0)

            # center = (module_w / 2, module_h / 2)
            # M = cv2.getRotationMatrix2D(center, angle, scale)
            # rotated = cv2.warpAffine(module, M, (module_w, module_h))
            # # cv2.namedWindow("test", 0)
            # # cv2.imshow("test", rotated)
            # # cv2.waitKey(0)
            # img_name = os.path.basename(img_path)
            # cv2.imwrite(os.path.join(new_folder_path, img_name), rotated)

        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(os.path.join(new_dir, img_name), img)
        # if '_GAN' not in img_name:
        #     pass
        #     # os.remove(img_path)
        #     num += 1

        # if '_o.jpg' in img_name:
        #     print(img_path)
        #     num += 1
        #     pass
        # elif '_sh.jpg' in img_name:
        #     if '_GAN_sh.jpg' not in img_name:
        #         print(img_path)
        #         num2 += 1
        # img_name_new = img_name.replace("LUN", "LuN")
        # os.rename(img_name, img_name_new)
    print(num)
    print(num2)
def test_img():
    path = r"D:\data\Vietnam\GAN_100W_pre\8Q4421_GAN.jpg"
    img = cv2.imread(path)

    img_h = img.shape[0]
    img_w = img.shape[1]
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    max_gray0 = np.max(hist)
    elem1 = np.argmax(hist[0:])
    max_gray1 = float(hist[elem1])
    elem2 = np.argmax(hist[1:])
    max_gray2 = float(hist[elem2])
    if abs(elem2 - elem1) == 2:
        max_gray1 = max_gray1 + max_gray2
    # max_index = hist.index(max_gray0)
    if max_gray1 > img_h * img_w * 0.65:
        cv2.namedWindow("test", 0)
        cv2.imshow("test", img)
        cv2.waitKey(0)
def read_txt_img():
    txt_path = r"D:\data\UK\UK_train\train_clean\train_single_LP.txt"
    new_path = r"D:\data\UK\UK_train\train_clean\train_single_LP_just_UK.txt"
    new_list = []
    def show_img(filename):
        img = cv2.imread(filename)
        cv2.namedWindow("test", 0)
        cv2.imshow("test", img)
        cv2.waitKey(0)
    def check_label(img_path):
        with open(img_path[:-4] + '.txt', 'r') as label_txt:
            all_lines = label_txt.readlines()
            labels = [line.strip().split(' ') for line in all_lines]
            for anno in labels:
                label = anno[-2]
                label_up, label_bot = label.split("@")
                if len(label_bot) != 7:
                    return False
                if (label_bot[0] not in string.ascii_uppercase) or (label_bot[1] not in string.ascii_uppercase):
                    print(img_path)
                    # show_img(img_path)
                    return False
                if (label_bot[2] not in string.digits) or (label_bot[2] not in string.digits):
                    print(img_path)
                    # show_img(img_path)
                    return False
                if (label_bot[4] not in string.ascii_uppercase) or (label_bot[5] not in string.ascii_uppercase) or (
                        label_bot[6] not in string.ascii_uppercase):
                    print(img_path)
                    # show_img(img_path)
                    return False
            return True
    with open(new_path, 'w') as new_txt:
        with open(txt_path, "r") as txt:
            img_list = txt.readlines()
            for img_path in img_list:
                img_path = img_path.replace('\n', '')
                basename = os.path.basename(img_path)
                # print(img_path)
                # img = cv2.imread(img_path)
                basename = basename[:-4]
                if check_label(img_path):
                    new_txt.writelines(img_path+'\n')
                # shutil.copyfile(img_path,"{}\{}".format(new_folder, basename))
                # shutil.copyfile("{}.txt".format(img_path[:-4]), "{}\{}.txt".format(new_folder, basename[:-4]))
                # cv2.namedWindow("test", 0)
                # cv2.imshow("test", img)
                # cv2.waitKey(0)
            pass

def copy_folder():
    folder_path = r'D:\RD\Project\GeoLPDTraining-master\T_military_TD_test\T_military'
    times = 9
    for index in range(times):
        new_path = folder_path + str(index+2)
        shutil.copytree(folder_path, new_path)



def remove_file():
    path = r"D:\data\data_4GAN\EVS\France\GAN_result_old"
    shutil.rmtree(path)

def read_txt_do():
    txt_path = r"E:\data\Israel\test_clean2\error.txt"
    new_folder = r"E:\data\Israel\test_clean\special"
    num = 0
    with open(txt_path, 'r') as txt_file:
        img_list = txt_file.readlines()
    for img_path in img_list:
        img_path = img_path.replace('\n', '')
        file_name = os.path.basename(img_path)
        new_path = os.path.join(new_folder, file_name)
        label_file_path = img_path[:-3] + 'txt'
        new_label_file_path = os.path.join(new_folder, file_name[:-3] + 'txt')
        shutil.copyfile(img_path, new_path)
        shutil.copyfile(label_file_path, new_label_file_path)
        num+=1

    print("deal with " + str(num) + ' file' )

def taiwan_testbad_result_2_excel():
    def get_result(lines):
        result = ''
        for index, line in enumerate(lines):
            if index == 2:
                line = line.replace("\n", "")
                line = line.split(":")[-1]
                line = line.replace(", ", "\t")
                result+=line+"\t"
            else:
                line = line.replace("\n", "")
                line = line.split(": ")[-1]
                result+=line+"\t"
        return result




    folder = r"D:\data\Taiwan\testbad_result\8"
    txt_names = ["result_val.txt", "result_dirty.txt", "result_dirty_chung_long.txt", "result_OK2.txt", "result_T_military.txt",
                 "result_T_diplomat.txt", "result_ambassador.txt", "result_motorcycle.txt", "result_scooter.txt"]
    for name in txt_names:
        with open(os.path.join(folder, name), 'r', encoding="utf8") as txt_file:
            lines = txt_file.readlines()
            lines = lines[-7:-1]
            result = get_result(lines)
            print(result)

if __name__ == '__main__':
    # test()
    # test_img()
    # random_choice_from_GAN_pre()
    skan_folder()
    # read_txt_img()
    # check_rectangle()
    # remove_file()
    # copy_folder()
    # read_txt_do()
    # taiwan_testbad_result_2_excel()