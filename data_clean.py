# coding: UTF-8
import os
import sys
import glob
import argparse
import cv2
import shutil


def argparser():
    parser = argparse.ArgumentParser(prog='data clean')
    parser.add_argument('--data_dir', type=str, default=r'D:\data\Taiwan\Truck\dirty', required=False,
                        help='Path to input directory\n')
    parser.add_argument('--output_dir', type=str, default=r'D:\data\Taiwan\Truck_clean', required=False,
                        help='Output directory\n')
    parser.add_argument('--clean_mode', type=str, default='test',
                        required=False,
                        help='train(only delete cut difficult LP), test(delete any image with cut difficult)\n')
    return parser


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def loc_in_img_checker(label, img_shape):
    loc_x = label[::2]
    loc_y = label[1::2]
    if min(loc_x) < 0 or min(loc_y) < 0 or max(loc_x) > img_shape[1] or max(loc_y) > img_shape[0]:
        return False
    return True


def label_rule_checker(labels, img_shape):
    # labels = data_clean(labels,img_shape)
    #    labels = two_column_check(labels)
    new_labels = []
    try:
        for label in labels:
            if len(label) < 3:
                continue
            if int(label[2]) == 0:
                continue
            anno = list(map(int, label[0].split(',')))
            if len(anno) < 8:
                continue
            if (anno[4] + anno[5] - anno[0] - anno[1]) <= 0:
                continue
                # -------
            if int(label[2]) == 2 or int(label[2]) == 3:
                continue

            if (loc_in_img_checker(anno, img_shape)):
                new_labels.append(label)

        if len(new_labels) < 1:
            return False

    except:
        return False
    else:
        return new_labels
def label_rule_checker_test(labels, img_shape):
    # labels = data_clean(labels,img_shape)
    #    labels = two_column_check(labels)
    new_labels = []
    try:
        for label in labels:
            if len(label) < 3:
                continue
            if int(label[2]) == 0:
                continue
            anno = list(map(int, label[0].split(',')))
            if len(anno) < 8:
                continue
            if (anno[4] + anno[5] - anno[0] - anno[1]) <= 0:
                continue
                # -------
            if int(label[2]) == 2 or int(label[2]) == 3:
                continue

            if (loc_in_img_checker(anno, img_shape)):
                new_labels.append(label)

        if len(new_labels) < 1:
            return False

    except:
        return False
    else:
        return new_labels


def two_column_check(labels):
    return [label for label in labels if label[1].index('@') == 0]


def data_clean(labels, img_shape):
    return [anno_Modify(label, img_shape) for label in labels if len(label) == 3]


def anno_Modify(label, img_shape):
    anno = list(map(int, label[0].split(',')))
    anno = [max(min(img_shape[(x + 1) % 2], anno[x]), 0) for x in range(len(anno))]
    label[0] = ','.join(str(x) for x in anno)
    return label


# def main():
#     args = argparser().parse_args()
#     total = 0
#     noise = 0
#     badimg = 0
#     data_clean_list = []
#     savepath = args.output_dir
#     createFolder(savepath)
#     input_filename = args.data_dir
#     input_file_path = os.path.dirname(input_filename)
#     with open(input_filename, 'r') as input_txt:
#         lines = input_txt.readlines()
#     for line in lines:
#         line = "{}{}".format(input_file_path, line[1:-5])
#         basename = os.path.basename(line)[:-5]
#         img_folder = os.path.dirname(line)
#         img = cv2.imread('{}.jpg'.format(line))
#         if img is None:
#             print(line)
#             continue
#         txt_file_name = "{}.txt".format(line)
#         with open(txt_file_name, 'r', errors='ignore', encoding='UTF-8') as txt_file:
#             all_lines = txt_file.readlines()
#         labels = [line.strip().split(' ') for line in all_lines]
#
#         if args.clean_mode == "train":
#             labels = label_rule_checker(labels, img.shape)
#         else:
#             #args.clean_mode == "test"
#             labels = label_rule_checker_test(labels, img.shape)
#
#         if (labels):
#             modifyLabels = list(map(' '.join, labels))
#
#             now_folder = os.path.relpath(img_folder, input_file_path)
#             save_path = "{}\{}".format(args.output_dir, now_folder)
#             if not os.path.isdir(save_path):
#                 os.mkdir(save_path)
#             with open("{}\{}.txt".format(save_path, basename), 'w+') as outfile:
#                 outfile.write('\n'.join(modifyLabels))
#             shutil.copyfile('{}.jpg'.format(line), "{}\{}.jpg".format(save_path, basename))
#         else:
#             noise += 1
#             print(txt_file_name + ' has no license tag')
#     print('Total data : ' + str(total))
#     print('Noise data : ' + str(noise))
#     print('badimg data : ' + str(badimg))

def main():
    args = argparser().parse_args()
    total = 0
    noise = 0
    badimg = 0
    data_clean_list = []
    savepath = args.output_dir
    createFolder(savepath)
    input_filename = args.data_dir
    input_file_path = os.path.dirname(input_filename)
    lines = glob.glob(r"{}\*.jpg".format(input_filename))
    for line in lines:
        line = line[:-4]
        basename = os.path.basename(line)
        img_folder = os.path.dirname(line)
        img = cv2.imread('{}.jpg'.format(line))
        if img is None:
            print(line)
            continue
        txt_file_name = "{}.txt".format(line)
        with open(txt_file_name, 'r', errors='ignore', encoding='UTF-8') as txt_file:
            all_lines = txt_file.readlines()
        labels = [line.strip().split(' ') for line in all_lines]

        if args.clean_mode == "train":
            labels = label_rule_checker(labels, img.shape)
        else:
            # args.clean_mode == "test"
            labels = label_rule_checker_test(labels, img.shape)

        if (labels):
            modifyLabels = list(map(' '.join, labels))

            now_folder = os.path.relpath(img_folder, input_file_path)
            save_path = "{}\{}".format(args.output_dir, now_folder)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            with open("{}\{}.txt".format(save_path, basename), 'w+') as outfile:
                outfile.write('\n'.join(modifyLabels))
            shutil.copyfile('{}.jpg'.format(line), "{}\{}.jpg".format(save_path, basename))
        else:
            noise += 1
            print(txt_file_name + ' has no license tag')
    print('Total data : ' + str(total))
    print('Noise data : ' + str(noise))
    print('badimg data : ' + str(badimg))





if __name__ == '__main__':
    sys.exit(main() or 0)

