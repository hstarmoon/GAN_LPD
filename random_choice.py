import numpy as np
import random



def random_choice_WE(times):
    first_list = np.array(range(1, 39))
    second_list = np.array(range(1, 9))
    random.shuffle(first_list)
    random.shuffle(second_list)
    first_result = first_list[:6]
    first_result = np.sort(first_result)
    print("first = " + str(first_result))
    print("second = " + str(second_list[0]))

def random_choice_DA(times):

    for tmp in range(times):
        first_list = list(range(1, 49))
        result = random.sample(first_list, 6)
        # random.shuffle(first_list)
        # first_result = first_list[:6]
        result.sort()
        print("first = " + str(result))

def compare():
    input_number = [2, 20, 25, 30, 34, 37, 6]
    input_1_list = input_number[:-2]
    input_2_list = [input_number[-1]]
    set_input_1 = set(input_1_list)
    set_input_2 = set(input_2_list)

    txt_path = r'C:\Users\Albert\Desktop\txt\list.txt'
    with open(txt_path, 'r') as txt_file:
        lines = txt_file.readlines()
        for line in lines:
            line = line.split()
            real_1 = line[:-2]
            set_real_1 = set(real_1)
            real_2 = line[-1]
            set_real_2 = set(real_2)
            union_1 = set_input_1.intersection(set_real_1)
            union_2 = set_input_2.intersection(set_real_2)

if __name__ == '__main__':
    random_choice_WE(2)
    # random_choice_DA(4)
    # compare()
