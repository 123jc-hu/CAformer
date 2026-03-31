import os
import argparse
from main import seed_torch
from Data_Processing.make_dataset import load_preprocessed_data_2days, load_preprocessed_data_for_GIST
import numpy as np
import random


def generate_by_mtr(data):
    # 生成 masked temporal recongnition task 数据集
    masked_temporal_martrix = np.array([[0, 14], [15, 17], [18, 20], [21, 25], [26, 32], [33, 42],
                                        [43, 57], [58, 68], [69, 127]])
    [N, C, T] = data.shape

    # step1：扩充
    expand_data = []
    for i in range(N):
        for j in range(masked_temporal_martrix.shape[0]):
            expand_data.append(data[i])
    expand_data = np.array(expand_data)
    # step2: 修改
    for i in range(N * masked_temporal_martrix.shape[0]):
        sequence = i % masked_temporal_martrix.shape[0]
        for j in range(C):
            raw_signal = expand_data[i, j,
                         masked_temporal_martrix[sequence][0]: masked_temporal_martrix[sequence][1]]
            mean, var = np.mean(raw_signal), np.var(raw_signal)
            for m in range(masked_temporal_martrix[sequence][0], masked_temporal_martrix[sequence][1]):
                expand_data[i, j, m] = random.gauss(mean, var)
    x_mtr = np.array(expand_data)

    y_mtr = [[i for i in range(masked_temporal_martrix.shape[0])] for i in range(N)]
    y_mtr = np.array(y_mtr).reshape(N * masked_temporal_martrix.shape[0])

    return x_mtr, y_mtr


def generate_by_msr(dataset, data):
    # 生成 masked spatial recongnition task 数据集
    BiosemiRegion = [[0, 29, 1, 28],
                     [2, 5, 7, 6],
                     [3, 26, 30, 4, 25],
                     [27, 24, 22, 23],
                     [9, 11, 10],
                     [31, 8, 21, 12],
                     [20, 18, 19],
                     [13, 17, 14, 16, 15]]

    NeuralScanRegion = [[0, 1, 2, 3, 4, 59],
                        [5, 6, 15, 16, 14, 24, 25, 23],
                        [7, 8, 9, 10, 11, 17, 18, 19],
                        [12, 13, 20, 21, 22, 29, 30, 31],
                        [32, 33, 34, 41, 42, 43, 44, 58],
                        [26, 27, 28, 35, 36, 37, 45, 53],
                        [38, 39, 40, 46, 47, 48, 49, 61],
                        [50, 51, 52, 54, 55, 56, 57, 60]]
    region = NeuralScanRegion if dataset in ["CAS", "THU"] else BiosemiRegion
    # region = np.array(region)

    [N, C, T] = data.shape

    expand_data = []
    # step1: 扩充
    for i in range(N):
        for j in range(len(region)):
            expand_data.append(data[i])
    expand_data = np.array(expand_data)

    # step2: 修改
    for i in range(N * len(region)):
        sequence = i % len(region)
        for j in region[sequence]:
            raw_signal = expand_data[i, j, :]
            mean, var = np.mean(raw_signal), np.var(raw_signal)
            for m in range(T):
                expand_data[i, j, m] = random.gauss(mean, var)
        # self._draw(expand_data[i])
    x_msr = np.array(expand_data)

    y_msr = [[i for i in range(len(region))] for i in range(N)]
    y_msr = np.array(y_msr).reshape(N * len(region))

    return x_msr, y_msr


if __name__ == '__main__':
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 或者 ":16:8"

    parser = argparse.ArgumentParser(description='PyTorch Data Split and Save')
    parser.add_argument('--dataset', default='THU', type=str, help='dataset name')
    parser.add_argument('--sub_num', default=64, type=int, help='number of subjects')
    parser.add_argument("--sub_name", type=str, default="sub1", help="choose sub for each dataset")
    parser.add_argument("--n_fold", type=int, default=5, help="N fold cross-validation")
    parser.add_argument("--remove_num", type=int, default=0, help="number of removed non-targets")
    parser.add_argument("--random_seed", type=int, default=2024, help="choose random seed to help repeat")

    args = parser.parse_args()

    for sub_index in range(0, args.sub_num):
        seed_torch(args.random_seed)
        args.sub_name = f'sub{sub_index+1}'
        print(f'Processing {args.dataset}---{args.sub_name}...')

        # load data
        train_val_set, test_set = load_preprocessed_data_for_GIST(args) if args.dataset == 'GIST' else load_preprocessed_data_2days(args)

        train_data = [train_val_set[0][i] for i in range(len(train_val_set[0]))]
        train_label = [train_val_set[1][i] for i in range(len(train_val_set[1]))]
        fold_length = train_label[0].shape[0]
        train_data = np.concatenate(train_data, axis=0)
        train_label = np.concatenate(train_label, axis=0)
        test_data = test_set[0]
        test_label = test_set[1]

        # produce the data for MTCN
        train_data_mtr, train_label_mtr = generate_by_mtr(train_data)
        train_data_msr, train_label_msr = generate_by_msr(args.dataset, train_data)

        # save data
        save_folder = os.path.join(os.getcwd(), 'Dataset', args.dataset, 'fold5_data_for_MTCN')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, f'{args.sub_name}.npz')
        np.savez(save_path, train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label,
                 fold_length=fold_length, train_data_mtr=train_data_mtr, train_label_mtr=train_label_mtr,
                 train_data_msr=train_data_msr, train_label_msr=train_label_msr)

    print('finish!')
