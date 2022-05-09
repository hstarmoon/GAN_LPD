import numpy as np
import os, cv2, re, argparse, glob, math
from image_process import Random_Process, Image_Process

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l',
        '--image_list_path',
        type=str,
        help='Image list path',
        default=r'D:\data\Malaysia\train_clean_label')
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        default=r'D:\data\Malaysia\check_two\train_crop',
        help='output paths')
    parser.add_argument(
        '-r',
        '--rectify_image',
        action='store_true',
        help='Rectify license plate image',
        default=True)
    return parser.parse_args()

def cal_rec_area(p1,p2,p3,p4):
    # p1 = np.array([1,2,3])

    d12 = np.linalg.norm(p2 - p1)
    d23 = np.linalg.norm(p3 - p2)
    d34 = np.linalg.norm(p4 - p3)
    d41 = np.linalg.norm(p1 - p4)
    d24 = np.linalg.norm(p4 - p2)

    s1 = (d12+d41+d24)/2
    s2 = (d23+d34+d24)/2
    x = (s1-d12)
    x1 = s1*x
    area1 = (s1*(s1-d12)*(s1-d41)*(s1-d24))**0.5
    area2 = (s2*(s2-d23)*(s2-d34)*(s2-d24))**0.5
    area_rec = area1 + area2
    print(area_rec)
    return area_rec

def cross_square(rect1, rect2):
    width = min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])
    height = min(rect1[3], rect2[3]) - max(rect1[1], rect2[1])

    if width <= 0 or height <= 0:
        return 0
    else:
        return (width * height)
def crop_LP(image_list_path, ouput_path):
    list_root, list_filename = os.path.split(image_list_path)
    image_size = (94, 21)
    image_size = (211, 99)
    img_count = 0

    # with open(image_list_path, 'r') as image_list_file:
    #     image_list = np.array([os.path.join(list_root, tmp.strip()[2:]) for tmp in image_list_file.readlines()])
    image_list = glob.glob('{}\*.jpg'.format(image_list_path))

    for file in image_list:
        pre, ext = os.path.splitext(file)
        if ext.lower() not in ('.jpg', '.jpeg'):
            continue
        img_count += 1
        print(pre)

        img = cv2.imread(file, cv2.IMREAD_COLOR)
        anno_file_path = pre + ".txt"
        if not os.path.exists(anno_file_path):
            continue

        with open(anno_file_path, 'r') as anno_file:
            anno_info = np.array([re.split(' |, |,', tmp.strip()) for tmp in anno_file.readlines()])

            for anno in anno_info:
                if int(anno[-1]) != 1:  # 1: nomal license plate
                    continue

                label = anno[-2].upper()
                label_up, label_bot = label.split("@")


                # min_x = np.amin(np.int32([anno[0], anno[2], anno[4], anno[6]]))
                # max_x = np.amax(np.int32([anno[0], anno[2], anno[4], anno[6]]))
                # min_y = np.amin(np.int32([anno[1], anno[3], anno[5], anno[7]]))
                # max_y = np.amax(np.int32([anno[1], anno[3], anno[5], anno[7]]))
                # crop_image = img[min_y:max_y, min_x:max_x]
                # file_name = "{}\{}.jpg".format(ouput_path, label_bot)
                # cv2.imwrite(file_name, crop_image)

                if args.rectify_image:
                    pts1 = np.float32(
                        [[anno[0], anno[1]], [anno[2], anno[3]], [anno[4], anno[5]], [anno[6], anno[7]]])
                    pts2 = np.float32(
                        [[0, 0], [0, image_size[1]], [image_size[0], image_size[1]], [image_size[0], 0]])
                    matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    rect_img = cv2.warpPerspective(img, matrix, (image_size[0], image_size[1]))
                    rect_file_name = "{}\{}_{}_rec.jpg".format(ouput_path, img_count, label_bot)
                    cv2.imwrite(rect_file_name, rect_img)
def crop_LP_lp(image_list_path, ouput_path):
    list_root, list_filename = os.path.split(image_list_path)
    # image_size = (94, 21)
    image_size = (209, 88)
    img_count = 0

    # with open(image_list_path, 'r') as image_list_file:
    #     image_list = np.array([os.path.join(list_root, tmp.strip()[2:]) for tmp in image_list_file.readlines()])
    image_list = glob.glob('{}\*.jpg'.format(image_list_path))

    for file in image_list:
        pre, ext = os.path.splitext(file)
        if ext.lower() not in ('.jpg', '.jpeg'):
            continue
        img_count += 1
        print(pre)
        img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 3)
        # img = cv2.imread(file, cv2.IMREAD_COLOR)
        anno_file_path = pre + ".lp"
        if not os.path.exists(anno_file_path):
            continue

        with open(anno_file_path, 'r') as anno_file:
            anno_info = np.array([re.split(' |, |,', tmp.strip()) for tmp in anno_file.readlines()])

            for anno in anno_info:

                label = anno[-1].upper()
                label_up, label_bot = label.split("@")


                # min_x = np.amin(np.int32([anno[0], anno[2], anno[4], anno[6]]))
                # max_x = np.amax(np.int32([anno[0], anno[2], anno[4], anno[6]]))
                # min_y = np.amin(np.int32([anno[1], anno[3], anno[5], anno[7]]))
                # max_y = np.amax(np.int32([anno[1], anno[3], anno[5], anno[7]]))
                # crop_image = img[min_y:max_y, min_x:max_x]
                # file_name = "{}\{}.jpg".format(ouput_path, label_bot)
                # cv2.imwrite(file_name, crop_image)

                if args.rectify_image:
                    pts1 = np.float32(
                        [[anno[0], anno[1]], [anno[2], anno[3]], [anno[4], anno[5]], [anno[6], anno[7]]])
                    pts2 = np.float32(
                        [[0, 0], [image_size[0], 0], [image_size[0], image_size[1]], [0, image_size[1]]])
                    matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    rect_img = cv2.warpPerspective(img, matrix, (image_size[0], image_size[1]))
                    rect_file_name = "{}\{}_{}_rec.jpg".format(ouput_path, img_count, label_bot)
                    cv2.imwrite(rect_file_name, rect_img)

def crop_LP_txt(image_list_path, ouput_path, rec):
    list_root, list_filename = os.path.split(image_list_path)
    image_size = (94, 21)
    # image_size = (256, 76)
    count = 0

    image_list = glob.glob(os.path.join(image_list_path, "*.jpg"))
    # with open(image_list_path, 'r') as image_list_file:
    #     image_list = np.array([os.path.join(list_root, tmp.strip()[2:]) for tmp in image_list_file.readlines()])
    for file in image_list:
        pre, ext = os.path.splitext(file)
        if ext.lower() not in ('.jpg', '.jpeg'):
            continue

        # print(pre)

        img = cv2.imread(file, cv2.IMREAD_COLOR)
        anno_file_path = pre + ".txt"
        if not os.path.exists(anno_file_path):
            continue

        with open(anno_file_path, 'r') as anno_file:
            anno_info = np.array([re.split(' |, |,', tmp.strip()) for tmp in anno_file.readlines()])

            for anno in anno_info:
                if int(anno[-1]) != 1:  # 1: nomal license plate
                    continue

                label = anno[-2].upper()
                label_up, label_bot = label.split("@")
                if label_up:
                    new_label = label_up + "-" + label_bot
                else:
                    new_label = label_bot
                # min_x = np.amin(np.int32([anno[0], anno[2], anno[4], anno[6]]))
                # max_x = np.amax(np.int32([anno[0], anno[2], anno[4], anno[6]]))
                # min_y = np.amin(np.int32([anno[1], anno[3], anno[5], anno[7]]))
                # max_y = np.amax(np.int32([anno[1], anno[3], anno[5], anno[7]]))
                # crop_image = img[min_y:max_y, min_x:max_x]
                # file_name = "{}\{}.jpg".format(ouput_path, label_bot)
                # cv2.imwrite(file_name, crop_image)

                if args.rectify_image:
                    p1 = np.array([int(anno[0]), int(anno[1])])
                    p2 = np.array([int(anno[2]), int(anno[3])])
                    p3 = np.array([int(anno[4]), int(anno[5])])
                    p4 = np.array([int(anno[6]), int(anno[7])])
                    # area_rec = cal_rec_area( p1, p2, p3, p4)
                    # print(area_rec)
                    # if area_rec < 6000:
                    #     continue
                    count +=1

                    pts1 = np.float32(
                        [[anno[0], anno[1]], [anno[2], anno[3]], [anno[4], anno[5]], [anno[6], anno[7]]])
                    pts2 = np.float32(
                        [[0, 0], [0, image_size[1]], [image_size[0], image_size[1]], [image_size[0], 0]])
                    matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    rect_img = cv2.warpPerspective(img, matrix, (image_size[0], image_size[1]))
                    # img_padding = Image_Process().image_padding(rect_img, 256, 256, [0, 0, 0])
                    rect_file_name = "{}\{}_rec_{}.jpg".format(ouput_path, new_label, count)
                    cv2.imwrite(rect_file_name, rect_img)
            # print(count)
def crop_LP_txt_background(image_list_path, ouput_path, rec):
    list_root, list_filename = os.path.split(image_list_path)
    # image_size = (94, 21)
    image_size = (211, 99)

    with open(image_list_path, 'r') as image_list_file:
        image_list = np.array([os.path.join(list_root, tmp.strip()[2:]) for tmp in image_list_file.readlines()])
    for file in image_list:
        pre, ext = os.path.splitext(file)
        if ext.lower() not in ('.jpg', '.jpeg'):
            continue

        print(pre)

        img = cv2.imread(file, cv2.IMREAD_COLOR)
        anno_file_path = pre + ".txt"
        if not os.path.exists(anno_file_path):
            continue

        with open(anno_file_path, 'r') as anno_file:
            anno_info = np.array([re.split(' |, |,', tmp.strip()) for tmp in anno_file.readlines()])

            for anno in anno_info:
                if int(anno[-1]) != 1:  # 1: nomal license plate
                    continue

                label = anno[-2].upper()
                label_up, label_bot = label.split("@")

                # min_x = np.amin(np.int32([anno[0], anno[2], anno[4], anno[6]]))
                # max_x = np.amax(np.int32([anno[0], anno[2], anno[4], anno[6]]))
                # min_y = np.amin(np.int32([anno[1], anno[3], anno[5], anno[7]]))
                # max_y = np.amax(np.int32([anno[1], anno[3], anno[5], anno[7]]))
                # crop_image = img[min_y:max_y, min_x:max_x]
                # file_name = "{}\{}.jpg".format(ouput_path, label_bot)
                # cv2.imwrite(file_name, crop_image)
                # np.rad2deg
                # angle = (np.arctan2(int(anno[5]) - int(anno[3]), int(anno[4]) - int(anno[2])))
                # new_shift = abs(15 / np.cos(angle))
                new_shift = math.sqrt((int(anno[4]) - int(anno[2]))**2 + (int(anno[5])-int(anno[3]))**2)
                new_shift = new_shift * 40/400
                # shift_in_rectangle = 10
                # shift_out_rectangle = 10
                for index in [0,1,2,7]:
                #     anno[index] = str(int(anno[index]) + np.random.randint(-shift_out_rectangle, shift_in_rectangle))
                    anno[index] = str(int(anno[index]) - new_shift)
                for index_tmp in [3,4,5,6]:
                #     anno[index_tmp] = str(int(anno[index_tmp]) + np.random.randint(-shift_in_rectangle, shift_out_rectangle))
                    anno[index_tmp] = str(int(anno[index_tmp]) + new_shift)
                # rand_move = [np.random.randint(-10, 10) for _ in range(8)]
                if args.rectify_image:
                    pts1 = np.float32(
                        [[anno[0], anno[1]], [anno[2], anno[3]], [anno[4], anno[5]], [anno[6], anno[7]]])
                    # pts1 += np.float32([[rand_move[0], rand_move[1]], [rand_move[2], rand_move[3]],
                    #                     [rand_move[4], rand_move[5]], [rand_move[6], rand_move[7]]])
                    pts2 = np.float32(
                        [[0, 0], [0, image_size[1]], [image_size[0], image_size[1]], [image_size[0], 0]])
                    matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    rect_img = cv2.warpPerspective(img, matrix, (image_size[0], image_size[1]))
                    rect_file_name = "{}\{}_rec.jpg".format(ouput_path, label_bot)
                    cv2.imwrite(rect_file_name, rect_img)

def get_real_label(real_label_file_path):
    list_label = []
    list_rec = []
    with open(real_label_file_path, 'r') as real_label_file:
        anno_info = np.array([re.split(' |, |,', tmp.strip()) for tmp in real_label_file.readlines()])

        for anno in anno_info:
            if anno[-1] != '1':
                a = 0
            min_x = np.amin(np.int32([anno[0], anno[2], anno[4], anno[6]]))
            max_x = np.amax(np.int32([anno[0], anno[2], anno[4], anno[6]]))
            min_y = np.amin(np.int32([anno[1], anno[3], anno[5], anno[7]]))
            max_y = np.amax(np.int32([anno[1], anno[3], anno[5], anno[7]]))
            label = anno[-2].upper()
            label_up, label_bot = label.split("@")
            list_label.append(label_bot)
            rec = [min_x, min_y, max_x, max_y]
            list_rec.append(rec)
        return list_label, list_rec
def crop_LP_system_result(image_list_path, ouput_path):
    list_root, list_filename = os.path.split(image_list_path)
    image_size = (256, 110)
    # image_size = (211, 99)
    label_bot = ''
    _, ext = os.path.splitext(image_list_path)
    if ext == ".txt":
        with open(image_list_path, 'r') as image_list_file:
            image_list = np.array([os.path.join(list_root, tmp.strip()[2:]) for tmp in image_list_file.readlines()])
    else:
        dir_list = glob.glob("{}/*".format(image_list_path))
        image_list = []
        for index, folder1 in enumerate(dir_list):
            if os.path.isdir(folder1):
                jpg_files = glob.glob("{}/*.jpg".format(folder1))
                image_list.extend(jpg_files)

    error = 0
    all_num=0
    for index_img, file in enumerate(image_list):
        pre, ext = os.path.splitext(file)
        if ext.lower() not in ('.jpg', '.jpeg'):
            continue

        # print(pre)

        img = cv2.imread(file, cv2.IMREAD_COLOR)
        anno_file_path = pre + ".lp"
        real_label_file_path = pre + ".txt"
        if not os.path.exists(anno_file_path):
            error += 1
            print(pre)
            continue


        with open(anno_file_path, 'r') as anno_file:
            # test = anno_file.readlines()
            anno_info = np.array([re.split(' |, |,', tmp.strip()) for tmp in anno_file.readlines()])
            all_num += len(anno_info)

            label_bot_list, rec_gt = get_real_label(real_label_file_path)
            for index, anno in enumerate(anno_info):
                label_bot = 'ERROR'
                # if index+1 > len(label_bot_list):
                #     label_bot = '*'
                # else:
                #     label_bot = label_bot_list[index]
                min_x = np.amin(np.int32([anno[0], anno[2], anno[4], anno[6]]))
                max_x = np.amax(np.int32([anno[0], anno[2], anno[4], anno[6]]))
                min_y = np.amin(np.int32([anno[1], anno[3], anno[5], anno[7]]))
                max_y = np.amax(np.int32([anno[1], anno[3], anno[5], anno[7]]))

                if min_x < 0: min_x = 0
                if min_y < 0: min_y = 0
                if max_x > img.shape[1]: min_x = img.shape[1]
                if max_y > img.shape[0]: min_x = img.shape[0]
                rec_lp = [min_x, min_y, max_x, max_y]
                if index_img == 654:
                    aaa= 1
                    pass
                for index_tmp in range(len(label_bot_list)):
                    cross_area_rate = cross_square(rec_gt[index_tmp], rec_lp) / ((rec_gt[index_tmp][2]-rec_gt[index_tmp][0])*(rec_gt[index_tmp][3]-rec_gt[index_tmp][1]))
                    if cross_area_rate > 0.7:
                        label_bot = label_bot_list[index_tmp]
                        del label_bot_list[index_tmp]
                        del rec_gt[index_tmp]
                        break


                # crop_image = img[min_y:max_y, min_x:max_x]
                # file_name = "{}\{}_{}.jpg".format(ouput_path, label_bot, index_img)
                # cv2.imwrite(file_name, crop_image)

                if args.rectify_image:
                    pts1 = np.float32(
                        [[anno[0], anno[1]], [anno[2], anno[3]], [anno[4], anno[5]], [anno[6], anno[7]]])
                    pts2 = np.float32(
                        [[0, 0], [image_size[0], 0], [image_size[0], image_size[1]], [0, image_size[1]]])
                    matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    rect_img = cv2.warpPerspective(img, matrix, (image_size[0], image_size[1]))
                    rect_file_name = "{}\{}_{}_rec.jpg".format(ouput_path, label_bot,index_img)
                    cv2.imwrite(rect_file_name, rect_img)
                    a=0

    print(error)
    print(all_num)
def crop_LP_testbed(image_list_path, ouput_path):
    list_root, list_filename = os.path.split(image_list_path)
    image_size = (256, 63)
    # image_size = (211, 99)
    label_bot = ''
    image_list = glob.glob('{}\*.jpg'.format(image_list_path))
    # image_list = glob.glob(r'{}\*.jpg'.format(image_list_path))
    error = 0
    img_count = 0
    all_num = 0
    for index_img, file in enumerate(image_list):
        print(index_img)
        pre, ext = os.path.splitext(file)
        if ext.lower() not in ('.jpg', '.jpeg'):
            continue
        img_count += 1
        # print(pre)

        img = cv2.imread(file, cv2.IMREAD_COLOR)
        anno_file_path = pre + ".lp"
        real_label_file_path = pre + ".txt"
        if not os.path.exists(anno_file_path):
            error += 1
            print(pre)
            continue

        with open(anno_file_path, 'r') as anno_file:
            # test = anno_file.readlines()
            anno_info = np.array([re.split(' |, |,', tmp.strip()) for tmp in anno_file.readlines()])
            all_num += len(anno_info)

            for index, anno in enumerate(anno_info):
                label_bot_list, rec_gt = get_real_label(real_label_file_path)
                label_bot_gt = 'None'
                _, label_bot_predict = anno[-1].split('@')
                # if index+1 > len(label_bot_list):
                #     label_bot = '*'
                # else:
                #     label_bot = label_bot_list[index]
                min_x = np.amin(np.int32([anno[0], anno[2], anno[4], anno[6]]))
                max_x = np.amax(np.int32([anno[0], anno[2], anno[4], anno[6]]))
                min_y = np.amin(np.int32([anno[1], anno[3], anno[5], anno[7]]))
                max_y = np.amax(np.int32([anno[1], anno[3], anno[5], anno[7]]))

                if min_x < 0: min_x = 0
                if min_y < 0: min_y = 0
                if max_x > img.shape[1]: min_x = img.shape[1]
                if max_y > img.shape[0]: min_x = img.shape[0]
                rec_lp = [min_x, min_y, max_x, max_y]
                for index_tmp in range(len(label_bot_list)):
                    cross_area_rate = cross_square(rec_gt[index_tmp], rec_lp) / ((rec_gt[index_tmp][2]-rec_gt[index_tmp][0])*(rec_gt[index_tmp][3]-rec_gt[index_tmp][1]))
                    if cross_area_rate > 0.5:
                        label_bot_gt = label_bot_list[index_tmp]
                        del label_bot_list[index_tmp]

                # crop_image = img[min_y:max_y, min_x:max_x]
                # file_name = "{}\{}_{}.jpg".format(ouput_path, label_bot, index_img)
                # cv2.imwrite(file_name, crop_image)

                if args.rectify_image:
                    pts1 = np.float32(
                        [[anno[0], anno[1]], [anno[2], anno[3]], [anno[4], anno[5]], [anno[6], anno[7]]])
                    pts2 = np.float32(
                        [[0, 0], [image_size[0], 0], [image_size[0], image_size[1]], [0, image_size[1]]])
                    matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    rect_img = cv2.warpPerspective(img, matrix, (image_size[0], image_size[1]))
                    if label_bot_predict != label_bot_gt:
                        lpr_result = "fail"
                    else:
                        lpr_result = "OK"
                    rect_file_name = "{}\{}_{}_{}_{}_rec.jpg".format(ouput_path, img_count, label_bot_predict, label_bot_gt, lpr_result, index_img)
                    cv2.imwrite(rect_file_name, rect_img)
                    a = 0

    print(error)
    print(all_num)
def test_repeat_index_system_result():

    image_list = glob.glob('{}\*.jpg'.format(image_list_path))

if __name__ == '__main__':
    args = init_args()
    # _, ext = path.splitext(args.image_list_path)
    # if ext == ".txt":
    #
    # crop_LP_system_result(image_list_path=args.image_list_path, ouput_path=args.output_dir)
    # crop_LP_testbed(image_list_path=args.image_list_path, ouput_path=args.output_dir)
    crop_LP_txt(image_list_path=args.image_list_path, ouput_path=args.output_dir, rec=args.rectify_image)
    # crop_LP_txt_background(image_list_path=args.image_list_path, ouput_path=args.output_dir, rec=args.rectify_image)
    # crop_LP(image_list_path=args.image_list_path, ouput_path=args.output_dir)
    # crop_LP_lp(image_list_path=args.image_list_path, ouput_path=args.output_dir)
