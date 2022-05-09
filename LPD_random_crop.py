
import argparse,cv2,random, os, shutil
import numpy as np


def argparser():
    parser = argparse.ArgumentParser(prog='data clean')
    parser.add_argument('--data_dir',   type = str, default = r'D:\data\Malaysia\tmp\train.txt', required=False,
                        help='Path to input data\n' )
    parser.add_argument('--out_dir', type=str, default=r'D:\data\Malaysia\tmp_LPD',
                        required=False,
                        help='Path to input data\n')
    parser.add_argument('--num_crop', type=int, default = 1,
                        required=False,
                        help='Path to input data\n')
    parser.add_argument('--offset', default=30, type=int,
                        help='Extend box width and height,unit is percentage\n')

    return parser

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def LPD_random_crop(img_h, img_w, min_x, min_y, max_x, max_y, bios):
    new_min_x = 0
    new_min_y = 0
    new_max_x = img_w
    new_max_y = img_h
    if (min_x - bios) > 0:
        new_min_x = random.randint(0, (min_x - bios))
    if (min_y - bios) > 0:
        new_min_y = random.randint(0, (min_y - bios))
    if (max_x + bios) < img_w:
        new_max_x = random.randint((max_x + bios), img_w)
    if (max_y + bios) < img_h:
        new_max_y = random.randint((max_y + bios), img_h)
    return new_min_x, new_min_y, new_max_x, new_max_y

    pass
def get_new_LP_point(point0, LP_point):
    return [(LP_point[0] - point0[0]), (LP_point[1] - point0[1])]


def convert(img, box, expand=0):
    dw = 1./(img.shape[1])
    dh = 1./(img.shape[0])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    y = y*dh

    w = w*dw
    h = h*dh
    rect = [0, x, y, w, h]
    return rect

def main():

    args = argparser().parse_args()
    file_list_path = r"{}".format(args.data_dir)
    with open(file_list_path, 'r', encoding='utf8') as jpg_path_list_file:
        file_path_lists = jpg_path_list_file.readlines()
    for jpg_file_path in file_path_lists:
        jpg_file_path = jpg_file_path.strip()
        file_name = os.path.basename(jpg_file_path)[:-4]
        folder_name = os.path.basename(os.path.dirname(jpg_file_path))
        result_folder_path = os.path.join(args.out_dir, folder_name)
        if not os.path.exists(result_folder_path):
            createFolder(result_folder_path)
        LP_loc_list = []
        new_LP_loc_list = []
        LP_label_list = []
        new_LP_label_list = []
        LPD_label_list = []

        txt_filename = "{}.txt".format(jpg_file_path[0:-4])
        img = cv2.imdecode(np.fromfile(jpg_file_path, dtype=np.uint8), 3)
        img_h = img.shape[0]
        img_w = img.shape[1]
        print(txt_filename)
        with open(txt_filename, 'r') as label_file:
            lines = label_file.readlines()
        if len(lines) == 1:
            line = lines[0].split(' ')
            label = line[1] + ' ' + line[2]
            line = line[0]
            line = list(map(int, line.split(',')))
            LP_loc_list.append(line)
            LP_label_list.append(label)
            min_x = min(line[0], line[2], line[4], line[6])
            min_y = min(line[1], line[3], line[5], line[7])
            max_x = max(line[0], line[2], line[4], line[6])
            max_y = max(line[1], line[3], line[5], line[7])
            pass
        else:
            max_x = 0
            max_y = 0
            min_x = 999999
            min_y = 999999
            for line in lines:
                line = line.split(' ')
                label = line[1] + ' ' + line[2]
                line = line[0]
                line = list(map(int, line.split(',')))
                LP_loc_list.append(line)
                LP_label_list.append(label)
                min_x = min(min_x, line[0], line[2], line[4], line[6])
                min_y = min(min_y, line[1], line[3], line[5], line[7])
                max_x = max(max_x, line[0], line[2], line[4], line[6])
                max_y = max(max_y, line[1], line[3], line[5], line[7])
                pass
        for label in LP_loc_list:
            # label = label.split(',')
            LPD_label = convert(img, label)
            LPD_label = [str(i) for i in LPD_label]
            LPD_label = ' '.join(LPD_label)
            LPD_label_list.append(LPD_label)
        file_path = os.path.join(result_folder_path, file_name) + '.txt'
        with open(file_path, 'w+', encoding='utf8') as source_txt:
            source_txt.writelines(LPD_label_list)
        cv2.imencode('.jpg', img)[1].tofile(os.path.join(result_folder_path, file_name) + '.jpg')

        for index in range(args.num_crop):
            final_label = ''
            final_label_list = []
            new_min_x, new_min_y, new_max_x, new_max_y = LPD_random_crop(img_h, img_w, min_x, min_y, max_x, max_y, args.offset)
            # cv2.circle(img, (int(new_min_x), int(new_min_y)), 1, (255, 0, 255), 5)
            # cv2.circle(img, (int(new_max_x), int(new_max_y)), 1, (255, 0, 255), 5)
            # cv2.namedWindow("result", 0)
            # cv2.imshow("result", img)
            # cv2.waitKey(0)
            crop_img = img[new_min_y:new_max_y, new_min_x:new_max_x]
            for index_LP, LP_loc in enumerate(LP_loc_list):
                new_LP_loc = []
                for i in range(0,7,2):
                    new_LP_loc_tmp = get_new_LP_point([new_min_x, new_min_y],[LP_loc[i], LP_loc[i+1]])
                    # cv2.circle(crop_img, (int(new_LP_loc_tmp[0]), int(new_LP_loc_tmp[1])), 1, (255, 0, 255), 5)
                    new_LP_loc.extend(new_LP_loc_tmp)
                new_LP_loc = convert(crop_img, new_LP_loc)
                new_LP_loc = [str(i) for i in new_LP_loc]
                new_LP_loc = ' '.join(new_LP_loc)

                # modifyLabels = list(map(' '.join, new_LP_loc))

                # print(new_LP_loc)
                # final_label = new_LP_loc + ' ' + LP_label_list[index_LP]
                final_label_list.append(new_LP_loc)

            # print(new_LP_loc_list)
            # cv2.namedWindow("result")
            # cv2.imshow("result", crop_img)
            # cv2.waitKey(0)


            # str_loc = ", ".join()
            new_file_name = file_name + '_crop_' + str(index)
            result_txt_path = os.path.join(result_folder_path, new_file_name) + '.txt'
            result_img_path = os.path.join(result_folder_path, new_file_name) + '.jpg'

            with open(result_txt_path, 'w+', encoding='utf8') as result_txt:
                result_txt.writelines(final_label_list)
            cv2.imencode('.jpg', crop_img)[1].tofile(result_img_path)

            pass



if __name__ == '__main__':
    main()