import numpy as np
import os
from data_process import DataPreprocess, TrainTestSplit, load_data, save_data
import matplotlib.pyplot as plt
import argparse

dataset_dict = {
    "THU": r"..\Dataset\THU",
    "CAS": r"..\Dataset\CAS",
    "GIST": r"..\Dataset\GIST",
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parameters of data preprocess")
    parser.add_argument("--dataset", type=str, default="THU", help="choose dataset to preprocess")
    parser.add_argument("--sub_name", type=str, default="sub1A", help="choose sub data")
    parser.add_argument("--low_frequency", type=float, default=0.3, help="low frequency of bandpass filter")
    parser.add_argument("--high_frequency", type=float, default=40, help="high frequency of bandpass filter")
    parser.add_argument("--re_reference", type=str, default="Common Average Reference", help="re-reference mode")
    parser.add_argument("--new_fs", type=int, default=128, help="resample rate")

    args = parser.parse_args()

    for i in range(0, 55):
        args.dataset = "GIST"
        """1. 读取数据"""
        dateset_folder = dataset_dict[args.dataset]
        if args.dataset == "THU":
            args.sub_name = "sub{}A".format(i + 1)
            sub_num = i // 10
            sub_start = sub_num * 10 + 1
            sub_finish = sub_num * 10 + 10 if i < 60 else sub_num * 10 + 4
            dateset_folder = os.path.join(dateset_folder, f"S{sub_start}-S{sub_finish}.mat")
        elif args.dataset == "CAS":
            args.sub_name = "G{}D2".format(i + 1)
        elif args.dataset == "GIST":
            args.sub_name = "sub{}".format(i + 1)
        total_path = os.path.join(dateset_folder, f"{args.sub_name}.mat")
        sub_data, trigger_positions, class_labels, original_fs = load_data(total_path)
        num_target = np.sum(class_labels[0][-120:-1] == 1)
        """2. 预处理"""
        for i in range(len(sub_data)):
            # data_block=(C, T_all) trigger_block=(N,) label_block=(N,)
            data_block, trigger_block = sub_data[i], trigger_positions[i]
            delete_channels = [32, 42] if args.dataset == "THU" else None

            data_preprocess = DataPreprocess()

            # 去除坏道
            data_block = data_preprocess.delete_channels(data_block, delete_channels)

            # # 画图
            # data_split1 = data_preprocess.data_split(data_block, trigger_block, fs)
            # data_split1 = data_preprocess.baseline_remove(data_split1, fs, True)
            # target_before_filter = data_split1[np.where(class_labels[i]==1)[0][0]]
            # nontarget_before_filter = data_split1[np.where(class_labels[i]==2)[0][0]]
            # x = np.arange(original_fs)
            # plt.plot(x, target_before_filter[0], 'r', linestyle='-', label="target_before_filter")
            # plt.plot(x, nontarget_before_filter[0], 'b', linestyle='-', label="nontarget_before_filter")

            # 0.3-40Hz带通滤波
            data_block = data_preprocess.band_pass_filter(data_block, original_fs, args.low_frequency,
                                                          args.high_frequency)

            # 共平均参考
            data_block = data_preprocess.re_reference(data_block, args.re_reference)

            # -20ms~1000ms数据划分
            data_split = data_preprocess.data_split(data_block, trigger_block, original_fs)  # (N, C, 1.2T_origin)
            # 用-200ms~0ms数据去基线
            data_split = data_preprocess.baseline_remove(data_split, original_fs, if_b_r=True)  # (N, C, T_origin)

            # target_after_filter = data_split[np.where(class_labels[i] == 1)[0][0]]
            # tar_mean1 = target_after_filter[0].mean()
            # tar_std1 = target_after_filter[0].std()
            # nontarget_after_filter = data_split[np.where(class_labels[i] == 2)[0][0]]
            # plt.plot(x, target_after_filter[0], 'r', linestyle='-', label="target_after_filter")
            # plt.plot(x, nontarget_after_filter[0], 'b', linestyle='-', label="nontarget_after_filter")
            # plt.legend()
            # plt.show()

            # 重采样至128Hz
            if args.new_fs != original_fs:
                data_split = data_preprocess.resample_data_multiprocessing(data_split, original_fs,
                                                                           args.new_fs)  # (N, C, T_new)
            # 数据z-score标准化
            data_split = data_preprocess.scale_data(data_split)  # (N, C, T_new)

            # x_new = np.arange(args.new_fs)
            # target_after_std = data_split[np.where(class_labels[i] == 1)[0][0]]
            # nontarget_after_std = data_split[np.where(class_labels[i] == 2)[0][0]]
            # tar_mean2 = target_after_std[0].mean()
            # tar_std2 = target_after_std[0].std()
            # plt.plot(x_new, target_after_std[0], 'r', linestyle='-', label="target_after_filter")
            # plt.plot(x_new, nontarget_after_std[0], 'b', linestyle='-', label="nontarget_after_filter")
            # plt.legend()
            # plt.show()

            sub_data[i] = data_split

        """3. 保存数据"""
        folder_path = os.path.join(dateset_folder, "preprocessed")  # 获取文件夹路径
        sub_name = os.path.basename(total_path)  # sub1A.mat G1D1.mat sub1.mat
        sub_name_without_extension = os.path.splitext(sub_name)[0]  # sub1A G1D1 sub1

        # 如果是G1D1，需要获取到两个数字的信息
        if args.dataset == "CAS":
            characters = list(sub_name_without_extension)  # ["G", "1", "D", "1"]
            for j in range(2):
                sub_data_CAS = sub_data[j * 3:(j + 1) * 3]  # 取3个block为1个受试者数据
                sub_labels_CAS = class_labels[j * 3:(j + 1) * 3]
                sub_index = int(characters[1]) * 2 - 1
                if int(characters[-1]) == 1:
                    day_index = "A"
                else:
                    day_index = "B"
                sub_name_without_extension = f"sub{(sub_index + j)}" + day_index  # sub1A
                save_data(sub_data_CAS, sub_labels_CAS, folder_path, sub_name_without_extension)
        else:
            save_data(sub_data, class_labels, folder_path, sub_name_without_extension)

        print("finish")
