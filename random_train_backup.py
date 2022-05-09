import glob
from os import listdir,path,walk
import random
import argparse, shutil

def argparser():
    parser = argparse.ArgumentParser(prog='data clean')
    parser.add_argument('--data_dir',   type = str, default = r'D:\data\Taiwan\motorcycle', required=False,
                        help='Path to input data\n' )
    parser.add_argument('--mode', default = "i", type = str,
                        help='random with folder(f) or image file(i)\n')
    parser.add_argument('--train_rate', default=[0.9], type=list, help='the rate of train data')
    parser.add_argument('--copy_file', default=True, help='copy file to new path')
    parser.add_argument('--out_path', default=r'D:\data\Taiwan\m_train', type=str, help='out path')
    # parser.add_argument('--train_rate', default=0.8, type=int, help='the rate of train data')
    return parser

# dirlist = glob.glob(r'D:\data\Myanmar\GAN_LPD/*')
# outlist = (r'D:\data\Myanmar\GAN_LPD')

def random_img_from_txt(file_path, train_rate):
    with open(file_path, "r") as txt_file:
        txt_file_list = txt_file.read().splitlines()

        all_num = len(txt_file_list)
        offset = int(all_num * train_rate)
        random.shuffle(txt_file_list)
        train_list = txt_file_list[:offset]
        test_list = txt_file_list[offset:]
        file_path = path.split(file_path)[0]
        with open(path.join(file_path, "train.txt"), 'w') as train_file:
            train_file.write('\n'.join(train_list))
        with open(path.join(file_path, "test.txt"), 'w') as test_file:
            test_file.write('\n'.join(test_list))
        a= 0

def random_img(file_path, train_rate, out_path, copy_file):
    train_rate = train_rate[0]
    dirlist = glob.glob("{}/*".format(file_path))
    txt_file_list = []
    for index, folder1 in enumerate(dirlist):
        if path.isdir(folder1):
            jpg_files = glob.glob("{}/*.jpg".format(folder1))
            txt_file_list.extend(jpg_files)
    # out = all_num * train_rate
    all_num = len(txt_file_list)
    offset = int(all_num * train_rate)
    random.shuffle(txt_file_list)
    train_list = txt_file_list[:offset]
    test_list = txt_file_list[offset:]
    with open(path.join(file_path, "train.txt"), 'w') as train_file:
        train_file.write('\n'.join(train_list))
    with open(path.join(file_path, "test.txt"), 'w') as test_file:
        test_file.write('\n'.join(test_list))
    if copy_file:
        pass

def random_folder(file_path, train_rate, out_path, copy_file):
    num_train_rate = len(train_rate)
    train_list = []
    test_list = []
    val_list = []
    dirlist = []
    dirlist_tmp = glob.glob("{}/*".format(file_path))
    for index in dirlist_tmp:
        if path.isdir(index):
            dirlist.append(index)

    all_num = len(dirlist)
    offset = int(all_num * train_rate[0])
    random.shuffle(dirlist)
    train_folders = dirlist[:offset]
    tmp_folders = dirlist[offset:]
    if num_train_rate ==1:
        test_folders = tmp_folders
        for folder in train_folders:
            jpg_files = glob.glob("{}/*.jpg".format(folder))
            train_list.extend(jpg_files)
        for folder in test_folders:
            jpg_files = glob.glob("{}/*.jpg".format(folder))
            test_list.extend(jpg_files)

        with open("{}/train.txt".format(file_path), 'w', encoding="utf-8") as train_file:
            train_file.write('\n'.join(train_list))
        with open("{}/train_folder.txt".format(file_path), 'w', encoding="utf-8") as train_file:
            train_file.write('\n'.join(train_folders))
        with open("{}/test.txt".format(file_path), 'w', encoding="utf-8") as test_file:
            test_file.write('\n'.join(test_list))
        with open("{}/test_folder.txt".format(file_path), 'w', encoding="utf-8") as test_file:
            test_file.write('\n'.join(test_folders))

        if copy_file:
            for folder in train_folders:
                folder_name = path.basename(folder)
                shutil.copytree(folder, "{}/train/{}/".format(out_path, folder_name))
            for folder in test_folders:
                folder_name = path.basename(folder)
                shutil.copytree(folder, "{}/test/{}/".format(out_path, folder_name))
    else:#3個
        train_rate_tmp = train_rate[1] / (train_rate[1] + train_rate[2])
        offset2 = int((all_num-offset) * train_rate_tmp)
        test_folders = tmp_folders[:offset2]
        val_folders = tmp_folders[offset2:]
        for folder in train_folders:
            jpg_files = glob.glob("{}/*.jpg".format(folder))
            train_list.extend(jpg_files)
        for folder in test_folders:
            jpg_files = glob.glob("{}/*.jpg".format(folder))
            test_list.extend(jpg_files)
        for folder in val_folders:
            jpg_files = glob.glob("{}/*.jpg".format(folder))
            val_list.extend(jpg_files)

        with open("{}/train.txt".format(file_path), 'w', encoding="utf-8") as train_file:
            train_file.write('\n'.join(train_list))
        with open("{}/train_folder.txt".format(file_path), 'w', encoding="utf-8") as train_file:
            train_file.write('\n'.join(train_folders))
        with open("{}/test.txt".format(file_path), 'w', encoding="utf-8") as test_file:
            test_file.write('\n'.join(test_list))
        with open("{}/test_folder.txt".format(file_path), 'w', encoding="utf-8") as test_file:
            test_file.write('\n'.join(test_folders))
        with open("{}/val.txt".format(file_path), 'w', encoding="utf-8") as val_file:
            val_file.write('\n'.join(val_list))
        with open("{}/val_folder.txt".format(file_path), 'w', encoding="utf-8") as val_file:
            val_file.write('\n'.join(val_folders))

        if copy_file:
            for folder in train_folders:
                folder_name = path.basename(folder)
                shutil.copytree(folder, path.join(out_path, "train", folder_name))
            for folder in test_folders:
                folder_name = path.basename(folder)
                shutil.copytree(folder, path.join(out_path, "test", folder_name))
            for folder in val_folders:
                folder_name = path.basename(folder)
                shutil.copytree(folder, path.join(out_path, "val", folder_name))

def main():
    args =argparser().parse_args()
    file_path = r"{}".format(args.data_dir)
    mode = args.mode
    train_rate = args.train_rate
    _, ext = path.splitext(file_path)
    if ext == ".txt":
        random_img_from_txt(file_path, train_rate)
    else:
        if args.copy_file:
            out_path = args.out_path
        else:
            out_path = args.data_dir

        if mode == 'f':
            random_folder(file_path, train_rate, out_path, args.copy_file)
        else:
            random_img(file_path, train_rate, out_path, args.copy_file)

def scan_file_folder():
    input_path = r"D:\data\USA_CA\USA_CA_source_new\train"
    output_path = r""
    folder_names = listdir(input_path)
    base_folder = path.basename(input_path)
    with open("{}\{}.txt".format(input_path, base_folder), 'w') as txt_file:
        for line in folder_names:
            txt_file.write(line + '\n')
            # txt_file.writelines(folder_names+'\n')

    all_folder_names = []

    for index in range(1, 1114):
        name = "CA" + str(index).zfill(4)
        all_folder_names.append(name)
    diffirence = list(set(all_folder_names).difference(set(folder_names)))
    diffirence.sort()
    with open(r"{}\test.txt".format(input_path), 'w') as txt_file:
        for line in diffirence:
            txt_file.write(line + '\n')
    pass

def copy_folder_from_txt():
    txt_path = r"D:\data\EVS\German\new_data\Source\test_folder.txt"
    new_path = r"D:\data\EVS\German\new_data\test"
    source_path = r"D:\data\EVS\German\new_data\Source"
    with open(txt_path, 'r', encoding="utf-8") as txt_file:
        folder_paths = txt_file.readlines()
        for folder_path in folder_paths:
            folder_path = folder_path.strip('\n')
            folder = path.basename(folder_path)
            shutil.copytree(r"{}".format(folder_path), r"{}\test\{}".format(new_path, folder))



if __name__ == '__main__':
    main()
    # scan_file_folder()
    # copy_folder_from_txt()