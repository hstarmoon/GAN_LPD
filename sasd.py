import os, shutil

def main():
    path = r"D:\data\Vietnam\123"
    for index in range(1, 535):
        folder = "{}\Vietnam_{}".format(path, str(index).zfill(4))
        new_folder = "{}\Vietnam{}".format(path, str(index).zfill(4))
        os.rename(folder, new_folder)


def test():
    import cv2
    import numpy as np

    from matplotlib import pyplot as plt

    # read image
    img = cv2.imread(r'E:\data\people_count\te.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    channel = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channel[2], channel[2])

    cv2.merge(channel, hsv)
    cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB, img)


    # show image
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()