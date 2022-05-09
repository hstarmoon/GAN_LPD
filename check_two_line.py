import cv2
import numpy as np
import glob, os
import matplotlib.pyplot as plt

def check_hight_weight_rate(img_w, img_h):
    if (img_h < img_w):
        HightWidthRate = img_h / img_w
    else:
        HightWidthRate = img_w / img_h
    line_mod = 0
    if HightWidthRate < 0.3:
        line_mod = 1
    elif HightWidthRate > 0.7:
        line_mod = 2
    else:
        line_mod = 0
    print(HightWidthRate)
    return line_mod
def check_hight_weight_rate2(img_w, img_h):
    if (img_h < img_w):
        HightWidthRate = img_h / img_w
    else:
        HightWidthRate = img_w / img_h
    line_mod = 0
    if HightWidthRate < 0.55:
        line_mod = 1
    elif HightWidthRate >= 0.55:
        line_mod = 2
    else:
        line_mod = 0
    print(HightWidthRate)
    return line_mod
def check_two(img, anno, label_up, label_bot):
    if not ("-1" in anno):
        massage = ""
        anno_int = list(map(int, anno))
        img_w1 = CalcEuclideanDistance((anno_int[0], anno_int[1]), (anno_int[2], anno_int[3]))
        img_w2 = CalcEuclideanDistance((anno_int[4], anno_int[5]), (anno_int[6], anno_int[7]))
        img_h1 = CalcEuclideanDistance((anno_int[2], anno_int[3]), (anno_int[4], anno_int[5]))
        img_h2 = CalcEuclideanDistance((anno_int[6], anno_int[7]), (anno_int[0], anno_int[1]))
        img_w_avg = (img_w1 + img_w2) / 2
        img_h_avg = (img_h1 + img_h2) / 2
        line_mod = check_hight_weight_rate(img_w_avg, img_h_avg)
        if line_mod == 1:
            massage = "fist HW_rate"
            return line_mod, massage, img
        elif line_mod == 2:
            massage = "fist HW_rate"
            return line_mod, massage, img

        img_w_new = int(40 * (img_w_avg / img_h_avg))
        img_h_new = 40

        pts1 = np.float32([[anno[0], anno[1]], [anno[2], anno[3]], [anno[4], anno[5]], [anno[6], anno[7]]])
        pts2 = np.float32(
            [[0, 0], [img_w_new, 0], [img_w_new, img_h_new], [0, img_h_new]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        rect_img = cv2.warpPerspective(img, matrix, (img_w_new, img_h_new))
    else:
        print("3 point")
        remove_index = 0
        for index in range(len(anno)):
            if "-1" in anno[index]:
                remove_index = index
                break
        del anno[remove_index]
        del anno[remove_index]
        anno_int = list(map(int, anno))
        L1 = CalcEuclideanDistance((anno_int[0], anno_int[1]), (anno_int[2], anno_int[3]))
        L2 = CalcEuclideanDistance((anno_int[4], anno_int[5]), (anno_int[2], anno_int[3]))
        L3 = CalcEuclideanDistance((anno_int[0], anno_int[1]), (anno_int[4], anno_int[5]))
        list_L = [L1, L2, L3]
        list_L = sorted(list_L)
        line_mod = check_hight_weight_rate(list_L[1], list_L[0])
        if line_mod == 1:
            massage = "3 point fist HW_rate"
            return line_mod, massage, img
        elif line_mod == 2:
            massage = "3 point fist HW_rate"
            return line_mod, massage, img

        img_w_new = int(40 * (list_L[1] / list_L[0]))
        img_h_new = 40
        pts1 = np.float32([[anno[0], anno[1]], [anno[2], anno[3]], [anno[4], anno[5]]])
        pts2_list = [0, 0, img_w_new, 0, img_w_new, img_h_new, 0, img_h_new]
        del pts2_list[remove_index]
        del pts2_list[remove_index]
        pts2 = np.float32(
            [[pts2_list[0], pts2_list[1]], [pts2_list[2], pts2_list[3]], [pts2_list[4], pts2_list[5]]])
        matrix = cv2.getAffineTransform(pts1, pts2)
        rect_img = cv2.warpAffine(img, matrix, (img_w_new, img_h_new))


    img_h = rect_img.shape[0]
    img_w = rect_img.shape[1]

    module = np.zeros((img_h - 8, img_w - 8), dtype=np.uint8) + 255
    module = cv2.copyMakeBorder(module, 4, 4, 4, 4,
                                cv2.BORDER_CONSTANT, value=(0))
    # cv2.imwrite(r"D:\data\Malaysia\tmp\check_two/module.jpg", module)
    blur_img = cv2.GaussianBlur(rect_img, (9, 9), 0)
    # th6 = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 1)

    ret1, th1 = cv2.threshold(rect_img, 0, 255, cv2.THRESH_OTSU)
    # cv2.imwrite(r"D:\data\Malaysia\tmp\check_two\image_process\th1.jpg", th1)
    hist = cv2.calcHist([th1], [0], None, [2], [0, 256])
    if hist[0] > hist[1]:
        ret2, th2 = cv2.threshold(blur_img, 0, 255, cv2.THRESH_OTSU)
    else:
        ret2, th2 = cv2.threshold(blur_img, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    # plt.bar(range(1, 257), hist)
    # plt.plot(hist)
    # plt.show()
    k = np.ones((3, 3), np.uint8)  # kernel 這邊我們設定 5*5 的大小，作為捲積運算時的範圍尺寸
    erosion_img = cv2.erode(th2, k, iterations=1)  # erosion 運算
    opening_img = cv2.dilate(erosion_img, k, iterations=1)  # 對做完 erosion 的 img 再做 dilation 運算，就是 Opening
    add_img = cv2.addWeighted(module, 0.5, opening_img, 0.5, 0)
    _, add_img_new = cv2.threshold(add_img, 130, 255, cv2.THRESH_BINARY)

    img_labels, img_stats = cv2.connectedComponentsWithStats(add_img_new, 8)[1:3]
    area_count = img_stats[1:, 4]  # index 0 is background
    for i, obj_area in enumerate(area_count):
        if obj_area < 100:
            add_img_new[img_labels == i + 1] = 0
            continue
        height = img_stats[i + 1][3]
        width = img_stats[i + 1][2]
        percent_obj_rect = (height * width) / obj_area
        if (height < 10 and width > (0.2 * img_w)) or (width < 10 and height > (0.2 * img_h)):
            add_img_new[img_labels == i + 1] = 0
            continue
    k = np.ones((11, 11), np.uint8)
    dilate_img = cv2.dilate(add_img_new, k, iterations=1)
    close_img = cv2.erode(dilate_img, k, iterations=1)


    # cv2.imwrite(r"D:\data\Malaysia\tmp\check_two\image_process\source.jpg", rect_img)
    # cv2.imwrite(r"D:\data\Malaysia\tmp\check_two\image_process\blur_img.jpg", blur_img)
    # cv2.imwrite(r"D:\data\Malaysia\tmp\check_two\image_process\th2.jpg", th2)
    # cv2.imwrite(r"D:\data\Malaysia\tmp\check_two\image_process\opening_img.jpg", opening_img)
    # cv2.imwrite(r"D:\data\Malaysia\tmp\check_two\image_process\add_img.jpg", add_img)
    # cv2.imwrite(r"D:\data\Malaysia\tmp\check_two\image_process\add_img_new.jpg", add_img_new)
    # cv2.imwrite(r"D:\data\Malaysia\tmp\check_two\image_process\close_img.jpg", close_img)
    # cv2.imwrite(r"D:\data\Malaysia\tmp\check_two\image_process\module.jpg", module)
    contours, hierarchy = cv2.findContours(close_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour = []
    concat_img = cv2.hconcat([rect_img, th2, opening_img, add_img, add_img_new, close_img])
    if len(contours) > 1:
        print("len = ")
        print(len(contours))
        for cont in contours:
            contour.extend(cont)
        contours = np.array(contour)
    elif len(contours) < 1:
        print("errrrrrrrrrrror")
        massage = "no license plate"
        line_mod = 1
        return line_mod, massage, concat_img
        # cv2.namedWindow("concat_img", 1)
        # cv2.imshow("concat_img", concat_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        contours = contours[0]
    x, y, w, h = cv2.boundingRect(contours)
    print([x, y, w, h])
    cv2.rectangle(close_img, (x, y), (x+w, y+h), 125, 2)
    concat_img = cv2.hconcat([rect_img, th2, opening_img, add_img, add_img_new, close_img])

    up_distance = y
    down_distance = img_h - h - y
    if (h < img_h * 0.5) and (abs(up_distance - down_distance) > 0.4 * img_h):
        massage = "to far"
        return 2, massage, concat_img
    if w * h < img_w * img_h * 0.1:
        massage = "to small"
        print(w * h)
        print(img_w * img_h)
        return 1, massage, concat_img
    line_mod = check_hight_weight_rate2(w, h)
    if line_mod != 0:
        massage = "second HW_rate"
        return line_mod, massage, concat_img


    # cv2.namedWindow("concat_img", 1)
    # cv2.imshow("concat_img", concat_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("!!!!!!!")
    massage = "no"
    return 0, massage, concat_img




def image_padding(self, image, target_height=64, target_width=64, back_ground_color=[0, 0, 0]):
    '''
    :param image: 輸入圖片
    :param target_height: 補0後高度
    :param target_width: 補0後寬度
    :return: 邊緣補0後的圖
    '''
    height = image.shape[0]
    width = image.shape[1]
    padding_height = (target_height - height) // 2
    padding_width = (target_width - width) // 2
    padding_height = 0 if padding_height < 0 else padding_height
    padding_width = 0 if padding_width < 0 else padding_width
    image = cv2.copyMakeBorder(image, padding_height, padding_height, padding_width, padding_width,
                               cv2.BORDER_CONSTANT, value=back_ground_color)
    image = cv2.resize(image, (target_width, target_height))
    return image


def CalcEuclideanDistance(point1, point2):
    vec1 = np.array(point1)
    vec2 = np.array(point2)
    distance = np.linalg.norm(vec1 - vec2)
    return distance


# 計算第四個點
def CalcFourthPoint(point1, point2, point3):  # pint3為A點
    D = (point1[0] + point2[0] - point3[0], point1[1] + point2[1] - point3[1])
    return D


# 三點構成一個三角形，利用兩點之間的距離，判斷鄰邊AB和AC,利用向量法以及平行四邊形法則，可以求得第四個點D
def JudgeBeveling(point1, point2, point3):
    dist1 = CalcEuclideanDistance(point1, point2)
    dist2 = CalcEuclideanDistance(point1, point3)
    dist3 = CalcEuclideanDistance(point2, point3)
    dist = [dist1, dist2, dist3]
    max_dist = dist.index(max(dist))
    if max_dist == 0:
        D = CalcFourthPoint(point1, point2, point3)
    elif max_dist == 1:
        D = CalcFourthPoint(point1, point3, point2)
    else:
        D = CalcFourthPoint(point2, point3, point1)
    return D



def main():


    file_path = r"D:\data\Malaysia\check_two\train_twoline_fail"
    # file_path = r"D:\data\Malaysia\testbad_result\fail"
    out_path = r"D:\data\Malaysia\tmp\check_two\to_small"
    dirlist = glob.glob("{}/*".format(file_path))
    image_size = (100, 40)
    img_file_list = []
    for index, folder1 in enumerate(dirlist):
        if os.path.isdir(folder1):
            jpg_files = glob.glob("{}/*.jpg".format(folder1))
            img_file_list.extend(jpg_files)
    for file in img_file_list:
        file_name = os.path.basename(file)

        img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 0)

        lp_filename = "{}.lp".format(file[0:-4])
        txt_filename = "{}.txt".format(file[0:-4])
        # with open(txt_filename, 'r') as txt_file:
        #     lines_txt = txt_file.readlines()
        with open(lp_filename, 'r') as label_file:
            lines = label_file.readlines()
        for line in lines:
            line = line.split()
            if len(line) < 2:
                continue

            anno = line[0]
            anno = anno.split(',')

            label = line[1]
            label_up, label_bot = label.split("@")
            fail_rec = 0
            two_line, massage, result_img = check_two(img, anno, label_up, label_bot)
            # cv2.namedWindow("result_img", 1)
            # cv2.imshow("result_img", result_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.imwrite(r"{}/{}.jpg".format(out_path, file_name), result_img)
            if "to small" in massage:
                pass
                # cv2.imwrite(r"{}/{}.jpg".format(out_path, file_name), result_img)
            if two_line ==2:
                two_line = True
            elif two_line == 1:
                two_line = False
            else:
                two_line = False
                massage = "guess"
            if label_up:
                if two_line:
                    print("two line")
                    print(massage)
                elif two_line:
                    print("false two line")
                    print(massage)
                    print(line)
                    cv2.namedWindow("result_img", 1)
                    cv2.imshow("result_img", result_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            else:
                if two_line:
                    print("false one line")
                    print(massage)
                    print(line)
                    cv2.namedWindow("result_img", 1)
                    cv2.imshow("result_img", result_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("one line")
                    print(massage)

if __name__ == '__main__':
    main()