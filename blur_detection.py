# coding=utf-8
# 导入相应的python包
from imutils import paths
import argparse
import cv2, glob, os, re
import numpy as np

def variance_of_laplacian(image):
	'''
    计算图像的laplacian响应的方差值
    '''
	return cv2.Laplacian(image, cv2.CV_64F).var()


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l',
        '--image_list_path',
        type=str,
        help='Image list path',
        default=r'D:\data\Taiwan\blur\blur_problem\source')
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        help='设置模糊阈值',
        default=150)
    parser.add_argument(
        '--save_path',
        type=str,
        help='save path',
        default=r'D:\data\Taiwan\blur\blur_problem\result\222')
    return parser.parse_args()


def sp_noiseImg(img_file1, prob):  # 同时加杂乱(RGB单噪声)RGB图噪声 prob:噪声占比
    image = np.array(img_file1, dtype=float)
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    # prob = 0.05 #噪声占比 已经比较明显了 >0.1 严重影响画质
    NoiseImg = image.copy()
    NoiseNum = int(prob * image.shape[0] * image.shape[1])
    print("椒盐噪声图")
    print("噪声数量=", NoiseNum)
    for i in range(NoiseNum):
        rows = np.random.randint(0, image.shape[0] - 1)
        cols = np.random.randint(0, image.shape[1] - 1)
        channel = np.random.randint(0, 3)
        if np.random.randint(0, 2) == 0:  # 随机加盐或者加椒
            NoiseImg[rows, cols, channel] = 0
        else:
            NoiseImg[rows, cols, channel] = 255
    NoiseImg = np.array(NoiseImg, dtype=np.uint8)
    return NoiseImg
def sharpen(img, sigma=100):
    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)

    return usm
def custom_blur_demo(image):
    image = cv2.GaussianBlur(image, (3, 3), 15)
    # image = cv2.bilateralFilter(image, 7, 15, 15)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst
if __name__ == '__main__':
    args = init_args()
    img_count = 0
    image_size = (96, 41)
    image_list = []


    # with open(args.image_list_path, 'r') as image_list_file:
    #     image_list = np.array([os.path.join(list_root, tmp.strip()[2:]) for tmp in image_list_file.readlines()])
    image_list = glob.glob('{}\*.jpg'.format(args.image_list_path))
    img_folder_type = 1
    if len(image_list) < 1:
        img_folder_type = 2
        path_list = os.listdir(args.image_list_path)
        for sub_path in path_list:
            save_path = os.path.join(args.save_path, sub_path)
            if args.save_path:
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
            image_list.extend(glob.glob('{}\*.jpg'.format(os.path.join(args.image_list_path, sub_path))))
            a = 0
    # image_list = glob.glob('{}\*'.format(args.image_list_path))
    # if len(image_list)>0:
    #     for sub_list in image_list:
    #         image_list = glob.glob('{}\*.jpg'.format(args.image_list_path))
    # 遍历每一张图片


    for file in image_list:

        pre, ext = os.path.splitext(file)
        file_name = os.path.basename(pre)
        if img_folder_type == 2:
            folder = os.path.basename(os.path.dirname(pre))
            save_path = os.path.join(args.save_path, folder)
        elif img_folder_type == 1:
            save_path = args.save_path
        if ext.lower() not in ('.jpg', '.jpeg'):
            continue
        img_count += 1
        print(pre)

        img = cv2.imread(file, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h = img.shape[0]
        img_w = img.shape[1]
        anno_file_path = pre + ".lp"
        if not os.path.exists(anno_file_path):
            continue

        with open(anno_file_path, 'r') as anno_file:
            anno_info = np.array([re.split(' |, |,', tmp.strip()) for tmp in anno_file.readlines()])
            for index, anno in enumerate(anno_info):

                label = anno[-1].upper()
                label_up, label_bot = label.split("@")

                # min_x = np.amin(np.int32([anno[0], anno[2], anno[4], anno[6]]))
                # max_x = np.amax(np.int32([anno[0], anno[2], anno[4], anno[6]]))
                # min_y = np.amin(np.int32([anno[1], anno[3], anno[5], anno[7]]))
                # max_y = np.amax(np.int32([anno[1], anno[3], anno[5], anno[7]]))
                # crop_image = img[min_y:max_y, min_x:max_x]
                # file_name = "{}\{}.jpg".format(ouput_path, label_bot)
                # cv2.imwrite(file_name, crop_image)

                pts1 = np.float32(
                    [[anno[0], anno[1]], [anno[2], anno[3]], [anno[4], anno[5]], [anno[6], anno[7]]])
                pts2 = np.float32(
                    [[0, 0], [image_size[0], 0], [image_size[0], image_size[1]], [0, image_size[1]]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                rect_img = cv2.warpPerspective(img, matrix, (image_size[0], image_size[1]))
                rect_img = custom_blur_demo(rect_img)
                # rect_img = sharpen(rect_img, 100)
                # rect_img = sp_noiseImg(rect_img, 0.05)
                # cv2.namedWindow("LP", 0)
                # cv2.imshow("LP", rect_img)
                # cv2.waitKey(0)
                gray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)
                gray = sharpen(gray, 100)
                rect_file_name = "{}\{}_{}_rec.jpg".format(save_path, file_name, index)
                cv2.imwrite(rect_file_name, gray)

                # 计算灰度图片的方差
                fm = variance_of_laplacian(gray)
                text = "Not Blurry"

                # 设置输出的文字
                if fm < args.threshold:
                    text = "Blurry"

                # 显示结果
                print(fm)
                img_copy = img.copy()
                cv2.putText(img_copy, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                rect_img = cv2.copyMakeBorder(gray, 0, img_h - image_size[1], 0, 0,
                                              cv2.BORDER_CONSTANT, value=0)
                rect_img = cv2.cvtColor(rect_img, cv2.COLOR_GRAY2RGB)
                # cv2.namedWindow("LP", 0)
                # cv2.imshow("LP", rect_img)
                # cv2.waitKey(0)
                result = np.hstack([img_copy, rect_img])
                # cv2.namedWindow("Image", 0)
                # cv2.imshow("Image", result)
                # cv2.waitKey(0)
                rect_file_name = "{}\{}_{}.jpg".format(save_path, file_name, index)
                cv2.imwrite(rect_file_name, result)




