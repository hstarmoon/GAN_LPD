import glob
from os import listdir,path,walk
# import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse

ckeck_GT_rectangle = False
ckeck_pre_rectangle = True
LPD_base_dir = ''
TD_base_dir = ''
prediction_dir = ''
LPD_base_dir = r'D:\data\Malaysia\tmp_LPD\Mala0004'
# TD_base_dir = r"D:\data\all_TD\Euro0002"
# prediction_dir = r"D:\data\Malaysia\tmp\tmp1\Mala0001"
# base_dir = r"D:\data\Malaysia\tmp\tmp1_result\tmp3"
#
if LPD_base_dir != '':
    base_dir = LPD_base_dir
    mode = 'LPD'
elif TD_base_dir != '':
    base_dir = TD_base_dir
    mode = 'TD'
elif prediction_dir != '':
    base_dir = prediction_dir
    mode = 'pre'
else:
    mode = 'geo'



# mode = 'LPD'

def geo_2_rec(img, lines):
    img_high, img_with, _ = img.shape

    for line in lines:
        x_sum = 0
        y_sum = 0
        line = line.split()
        if len(line) < 3:
            continue
        class_label = line[2]
        label = line[1]
        label = label.split('@')[1]
        line = line[0]
        line = line.split(',')
        if len(line)<2:
            continue
        for index in range(0, 7, 2):
            loc = (int(line[index]), int(line[index+1]))
            x_sum += float(line[index])
            y_sum += float(line[index+1])
            cv2.circle(img, loc, 1, (255, 0, 255), 5)
        # cv2.putText(img, class_label,(int(x_sum/4), int(y_sum/4)), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 0, 0), 1)
        cv2.putText(img, label, (int(x_sum / 4), int(y_sum / 4)), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 0, 0), 3)
    cv2.namedWindow("result", 0)
    cv2.imshow("result", img)
    cv2.waitKey(0)

    a = 0

def pre_2_rec(img, lines, base_dir, file_name, ckeck_GT_rectangle, ckeck_pre_rectangle):
    def check_left_up(point):
        p_origin  = np.array([0, 0])
        loc_list = []
        L = []
        index_list = [0,1,2]
        L.append(np.linalg.norm(p_origin - point[0]))
        L.append(np.linalg.norm(p_origin - point[1]))
        L.append(np.linalg.norm(p_origin - point[2]))
        L.append(np.linalg.norm(p_origin - point[3]))
        for index in range(len(L)):
            if min(L) == L[index]:
                p0 = point[index]
                del point[index]
                L = []
                break
        L.append(np.linalg.norm(p0 - point[0]))
        L.append(np.linalg.norm(p0 - point[1]))
        L.append(np.linalg.norm(p0 - point[2]))
        for index in range(len(L)):
            if max(L) == L[index]:
                p2 = point[index]
                index_list.remove(index)
        if point[index_list[0]][0] < point[index_list[0]][1]:
            p1 = point[index_list[0]]
            p3 = point[index_list[1]]
        else:
            p1 = point[index_list[1]]
            p3 = point[index_list[0]]
        return p0, p1, p2, p3

    def get_new_loc(loc_list):
        point = []
        new_locs = []
        p0 = np.array(loc_list[0])
        p1 = np.array(loc_list[1])
        p2 = np.array(loc_list[2])
        p3 = np.array(loc_list[3])
        new_locs.append(p1 + (p3 - p2))
        new_locs.append(p0 + (p2 - p3))
        new_locs.append(p3 + (p1 - p0))
        new_locs.append(p2 + (p0 - p1))
        return new_locs
    lp_filename = "{}.lp".format(file_name)
    img_high, img_with, _ = img.shape

    for line in lines:
        loc_list = []
        line = line.split()
        if len(line) < 2:
            continue
        line = line[0]
        line = line.split(',')
        if len(line) < 2:
            continue
        for index in range(0, 7, 2):
            loc = (int(line[index]), int(line[index + 1]))
            cv2.circle(img, loc, 4, (255, 0, 0), 5)
            if ckeck_GT_rectangle:
                loc_list.append(loc)
        if ckeck_GT_rectangle:
            new_locs = get_new_loc(loc_list)
            for loc in new_locs:
                cv2.circle(img, tuple(loc), 4, (255, 255, 0), 1)


    with open(r"{}\{}".format(base_dir, lp_filename)) as lp_file:
        lp_lines = lp_file.readlines()

        for lp_line in lp_lines:
            predict_loc_list = []
            lp_line = lp_line.split()
            if len(lp_line) < 2:
                continue
            lp_label = lp_line[1]
            lp_line = lp_line[0]
            lp_line = lp_line.split(',')
            loc_x = int(min(lp_line[::2]))
            loc_y = int(min(lp_line[1::2]))
            if loc_x>1730: loc_x=1730
            if loc_y > 980: loc_y = 980
            if loc_y < 100: loc_y = 100
            if len(lp_line) < 2:
                continue
            for index in range(0, 7, 2):
                loc = (int(lp_line[index]), int(lp_line[index + 1]))
                cv2.circle(img, loc, 1, (255, 0, 255), 5)
                if ckeck_pre_rectangle:
                    predict_loc_list.append([loc[0], loc[1]])
            if ckeck_pre_rectangle:
                if len(predict_loc_list) != 4:
                    continue
                new_locs = get_new_loc(predict_loc_list)
                for loc in new_locs:
                    cv2.circle(img, tuple(loc), 4, (0, 255, 255), 2)
            cv2.putText(img, lp_label, (loc_x, loc_y-10), cv2.FONT_HERSHEY_DUPLEX,
                        1.2, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.namedWindow("result", 0)
    cv2.imshow("result", img)
    cv2.waitKey(0)
    aa = 0

def LPD_2_rec(img, lines, filename):
    img_high, img_with, _ = img.shape

    for line in lines:
        line = line.split()
        center_x = float(line[1]) * img_with
        center_y = float(line[2]) * img_high
        width_05 = float(line[3]) * img_with * 0.5
        high_05 = float(line[4]) * img_high * 0.5
        left_top = (int(center_x - width_05 + 0.5), int(center_y - high_05 + 0.5))
        right_down = (int(center_x + width_05 + 0.5), int(center_y + high_05 + 0.5))
        cv2.rectangle(img, left_top, right_down, (0, 0, 255), 2)

    cv2.namedWindow("result", 0)
    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.imwrite(r"{}_result\{}.jpg".format(base_dir, filename), img)

def TD_2_rec(img, lines):
    img_high, img_with, _ = img.shape
    # line = label_word.split()
    for line in lines:
        line = line.split()
        loc = (int(float(line[1])*img_with), int(float(line[2])*img_high))
        cv2.circle(img, loc, 1, (0, 0, 255), 4)

    cv2.namedWindow("result", 0)
    cv2.imshow("result", img)
    cv2.waitKey(0)
    # plt.imshow(img)
    # plt.show()
    a=0

# parser = argparse.ArgumentParser()
# parser.add_argument("file_path", type=str, default='', required=True,
#                     help="path of file",)
# parser.add_argument("train_rate", type=int, default=9, required=True,
#                     help="the rate of train data from 1 to 9",)
# args = parser.parse_args()

dirlist = glob.glob(r"{}\*.jpg".format(base_dir))
# outlist = (r'/root/workspace/GeoLPDTraining-master/GAN_LPD')

b_list = []
for file in dirlist:
    file_name = path.basename(file)

    img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 3)


    txt_filename = "{}.txt".format(file_name[0:-4])
    with open("{}\{}.txt".format(base_dir, file_name[0:-4]), 'r') as label_file:
        lines = label_file.readlines()
        print(txt_filename)
        if mode == "LPD":
           LPD_2_rec(img, lines, file_name[0:-4])
        elif mode== "TD":
            TD_2_rec(img, lines)
        elif mode == "pre":
            pre_2_rec(img, lines, base_dir, file_name[0:-4], ckeck_GT_rectangle, ckeck_pre_rectangle)
        else:
            geo_2_rec(img, lines)








