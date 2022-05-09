import os,glob,cv2,sys,shutil
data_dir = r'D:\data\Myanmar\Myanmar_ray\Myanmar.03\data\test_clean_tmp'
output_dir = r'D:\data\Myanmar\Myanmar_ray\Myanmar.03\data\test_clean_tmp2'

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
def main():
    total = 0
    noise = 0
    badimg = 0
    savepath = output_dir
    createFolder(savepath)
    try:
        for root, dirs, files in os.walk(data_dir):

            for txtfiles in glob.glob(os.path.join(root, "*.txt")):
                basename = os.path.basename(txtfiles)
                # imagepath = txtfiles[:-3] + ('jpg')
                shutil.copyfile(txtfiles[:-3] + ('jpg'),r"{}\{}.jpg".format(savepath, basename))
                shutil.copyfile(txtfiles[:-3] + ('txt'), r"{}\{}.txt".format(savepath, basename))
                shutil.copyfile(txtfiles[:-3] + ('lp'), r"{}\{}.lp".format(savepath, basename))

    except Exception as e:
        print(e)


if __name__ == '__main__':
    sys.exit(main() or 0)

