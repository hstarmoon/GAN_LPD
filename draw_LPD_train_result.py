import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(prog='plotTrainLoss')
parser.add_argument('--data_dir', type=str, default=r'D:\RD\tensorflow-yolo-v3\Malaysia\final_TD\TD_result.txt',
                    required=False)

args = parser.parse_args()
train_result_list = []
epoch_list = []
test_accuracy_list = []
file = args.data_dir
file_path = os.path.dirname(file)
file_basename = os.path.basename(file)
max_test_accuracy = 0
second_max_test_accuracy = 0
epoch_with_max_test_accuracy = 0

with open(file, 'r', encoding='utf8') as train_file:
    all_lines = train_file.readlines()

for line in all_lines:
    if "Loading weights from" in line:
        words_tmp = line.split(' ')[3]
        base_name = os.path.basename(words_tmp)
        base_name = base_name.split('.')[0]
        epoch_now = base_name.split('_')[-1]
        epoch_list.append(int(epoch_now))
    if "mean average precision (mAP" in line:
        words_tmp2 = line.split('=')[1]
        words_tmp2 = words_tmp2.split(',')[0]
        test_accuracy = float(words_tmp2)
        test_accuracy_list.append(test_accuracy)
        if test_accuracy > max_test_accuracy:
            second_max_test_accuracy = max_test_accuracy
            epoch_with_second_max_test_accuracy = epoch_with_max_test_accuracy
            max_test_accuracy = test_accuracy
            epoch_with_max_test_accuracy = epoch_now
        elif test_accuracy > second_max_test_accuracy:
            second_max_test_accuracy = test_accuracy
            epoch_with_second_max_test_accuracy = epoch_now

print('Retrieving data and plotting training loss graph...')
plt.style.use('bmh')

fig2 = plt.figure()
# plt.ylim((0.8, 0.99))
# plt.xticks(np.linspace(1000, epoch_now, tmp))
plt.xticks(fontsize=8, color="black", rotation=90)

for i in range(0, len(epoch_list)):
    # new_ticks = np.linspace(-1, 2, 5)
    # plt.yticks(new_ticks)
    plt.plot(epoch_list[i:i + 2], test_accuracy_list[i:i + 2], 'r.-')
result_name = "{}\{}_test.png".format(file_path, file_basename[:-4])
plt.xlabel('Batch Number')
plt.ylabel('test accuracy')
fig2.savefig(result_name, dpi=1000)

print("max test accuracy: " + str(max_test_accuracy))
print("epoch: " + str(epoch_with_max_test_accuracy))
print("second test accuracy: " + str(second_max_test_accuracy))
print("epoch: " + str(epoch_with_second_max_test_accuracy))

print('Done! Plot saved as training_loss_plot.png')