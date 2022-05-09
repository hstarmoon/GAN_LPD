import os, cv2, glob
import numpy as np


def check_rectangle(img, loc_list, thresh):
    distance = 0
    img_h = img.shape[0]
    img_w = img.shape[1]
    def get_new_loc(loc_list):
        new_locs = []
        p0, p1, p2, p3 = loc_list
        new_locs.append(p1 + (p3 - p2))
        new_locs.append(p0 + (p2 - p3))
        new_locs.append(p3 + (p1 - p0))
        new_locs.append(p2 + (p0 - p1))
        return new_locs

    new_locs = get_new_loc(loc_list)

    for index, _ in enumerate(new_locs):
        distance += np.linalg.norm(new_locs[index] - loc_list[index])

    if (distance / img_h) > thresh:
        print(distance / img_h)
        return False
    else:
        return True


def read_img():
    img_path = r"D:\data\test_rectangle\no"
    thresh = 0.06
    _, ext = os.path.splitext(img_path)
    if ext == ".txt":
        with open(img_path, "r") as txt_file:
            img_path_list = txt_file.read().splitlines()
    else:
        img_path_list = glob.glob(r"{}/*.jpg".format(img_path))

    for img_path in img_path_list:
        img = cv2.imread(img_path)
        with open("{}.lp".format(img_path[0:-4]), 'r') as label_file:
            lines = label_file.readlines()
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
                loc_list.append(np.array((int(line[index]), int(line[index + 1]))))
            if check_rectangle(img, loc_list, thresh):
                print("rectangle")
            else:
                print("not rectangle")


if __name__ == '__main__':
    read_img()