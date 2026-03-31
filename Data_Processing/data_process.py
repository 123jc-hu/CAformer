import numpy as np
from scipy import signal
from sklearn import preprocessing
import os
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor

"""
数据预处理总体分为6个步骤，不过不一定全用：
1. delete bad channels
2. filtering -> 0.1-40Hz
3. re-reference -> compute average reference
4. data split -> -200~1000ms
5. baseline remove -> -200~1000ms -> 0~1000ms
6. resampling -> 128Hz 由于重采样不是整数关系，只能先数据划分再重采样了
7. Normalization
"""


class DataPreprocess:
    def __init__(self):
        pass

    def delete_channels(self, data, channels_index=None):
        # data = (C, T_all)  channels_index = [n1, n2, ..., nk]
        if channels_index is not None:
            data = np.delete(data, channels_index, axis=0)
        return data

    def band_pass_filter(self, data, fs, freq_low=0.3, freq_high=40):
        # 带通滤波0.1-40Hz data = (C, T_all)
        wn = [freq_low * 2 / fs, freq_high * 2 / fs]
        [b, a] = signal.butter(4, wn, "bandpass")
        data = signal.filtfilt(b, a, data, axis=-1).astype(np.float32)
        return data

    def re_reference(self, data, mode="Common Average Reference"):
        # 使用共同平均重参考Common Average Reference或者平均乳突参考Average mastoids(M1M2，也可由TP9和TP10近似)
        # 不过两个数据集都没有TP9和TP10，所以还是算了
        # data = (C, T_all)
        reference_electrode = 0
        if mode == "Common Average Reference":
            reference_electrode = np.mean(data, axis=0)  # (1, T_all)
        elif mode == "Average mastoids":
            raise ValueError("Cannot do this operate! Please try Common Average Reference")
        re_referenced_data = data - reference_electrode
        return re_referenced_data

    def data_split(self, data, trigger_positions, fs):
        """
        提取-200~1000ms数据
        :param data: (C, T_all)，C为通道数，T_all为总时间点数
        :param trigger_positions: (data_num,)，每个元素表示一个触发点的位置
        :param fs: 采样率
        :return:
        """

        # 触发点前200ms的索引
        start_indices = trigger_positions - int(0.2 * fs)  # (data_num,)

        # 生成索引矩阵，每一行代表一个试验中每个时间点的索引
        idx_matrix = np.arange(int(1.2 * fs))[None, :]  # (1, 1.2*fs)

        # 将起始索引和索引矩阵相加，得到每个试验的时间窗口索引范围
        idx_ranges = start_indices[:, None] + idx_matrix  # (data_num, 1.2*fs)

        # 提取数据
        data_split = data[:, idx_ranges].transpose(1, 0, 2)  # (data_num, C, 1.2*fs)

        return data_split

    def baseline_remove(self, data_split, fs, if_b_r=True):
        # data_split = (N, C, T(-200~1000ms))
        data_baseline = data_split[:, :, 0:int(0.2*fs)]  # (N, C, T(-200~1000ms))
        data_baseline_mean = np.mean(data_baseline, axis=-1, keepdims=True)
        if if_b_r:
            data_baseline_remove = data_split[:, :, int(0.2*fs):] - data_baseline_mean
        else:
            data_baseline_remove = data_split[:, :, int(0.2*fs):] - 0
        return data_baseline_remove

    def resample_trial(self, trial_data, num_new_time):
        """重采样单个trial的函数"""
        return signal.resample(trial_data, num_new_time, axis=-1)

    def resample_data_multiprocessing(self, data_split, original_fs, new_fs):
        """重采样 data_split = (N, C, T(original_time_num)) -> (N, C, T(new_time_num))"""
        # 由于重采样后采样率与原采样率不是整数倍关系，所以只能先数据划分再采样了
        num_trials, _, num_original_time = data_split.shape
        num_new_time = int(num_original_time * new_fs / original_fs)

        # 创建一个ProcessPoolExecutor实例
        with ProcessPoolExecutor() as executor:
            # 使用executor.map来并行执行函数
            results = executor.map(
                self.resample_trial,
                data_split,  # 传递每个trial的数据
                [num_new_time] * num_trials  # 每个trial都使用相同的新时间点数
            )
        return np.array(list(results))

    def scale_data(self, data_split):
        # 标准化数据 data_split = (N, C, T)
        scaler = preprocessing.StandardScaler()
        for i in range(data_split.shape[0]):
            data_split[i] = scaler.fit_transform(data_split[i].transpose(1, 0)).transpose(1, 0)
        return data_split


class TrainTestSplit:
    """划分训练、验证、测试集"""
    def __init__(self):
        pass

    def nontarget_data_remove(self, n, data_split, labels):
        """
        去除目标前后n个非目标
        :param n: 去除目标位置前后n个非目标
        :param data_split: 分段好的数据(N, C, T)
        :param labels: 数据标签(N,) target=1, non-target=0
        :return: 去掉2n个非目标后的RSVP-EEG数据(N-2n, C, T)和对应标签(N-2n,)
        """
        target_indices = np.where(labels == 1)[0]
        nontarget_indices = np.where(labels == 2)[0]
        # 生成索引矩阵[-3, 3] astype(int)去了会报错，说索引不是整数
        idx_matrix = np.array([x for x in range(-n, n+1) if x != 0]).astype(int)[None, :]  # (1, 2n)
        # 获得要去除的非目标索引
        nontarget_remove_indices = target_indices[:, None] + idx_matrix
        # 去除2n个非目标索引
        nontarget_indices_new = np.setdiff1d(nontarget_indices, nontarget_remove_indices)

        # 新的数据集，先目标后非目标，未打乱
        data_split_new = np.concatenate([data_split[target_indices], data_split[nontarget_indices_new]], axis=0)
        labels_new = np.concatenate([np.ones(len(target_indices)), np.zeros(len(nontarget_indices_new))]).astype(np.int32)

        return data_split_new, labels_new

    def permute_data(self, data_split, labels=None):
        """
        打乱数据集
        :param data_split: 分段好的数据(N, C, T)
        :param labels: 数据对应标签
        :return: 打乱后的数据和标签
        """
        permutation_indices = np.random.permutation(data_split.shape[0])
        data_split = data_split[permutation_indices]
        if labels is not None:
            labels = labels[permutation_indices]

        return data_split, labels

    def split_train_validation_test(self, data_blocks, label_blocks, n_fold):
        """
        将数据集划分为训练集、验证集和测试集
        :param data_blocks: list 装着不同block的数据，其中block1用于训练，其余用于测试
        :param label_blocks: list 同上
        :param n_fold: int 为进行n折交叉验证对训练集分段
        :return: 训练集 验证集 测试集 ([x,y])
        """
        train_data, train_labels = np.concatenate(data_blocks[:-1], axis=0), np.concatenate(label_blocks[:-1], axis=0)
        test_set_x, test_set_y = data_blocks[-1], label_blocks[-1]
        # 将train_data进一步划分成训练集和验证集，共n份，对每一类数据单独划分
        target_num_index = np.where(train_labels == 1)[0]
        nontarget_num_index = np.where(train_labels == 0)[0]
        target_fold_num = len(target_num_index) // n_fold + 1
        # nontarget_fold_num = len(nontarget_num_index) // n_fold + 1
        train_target_data = train_data[target_num_index]
        train_nontarget_data = train_data[nontarget_num_index]
        # 打乱训练集
        train_target_data, _ = self.permute_data(train_target_data)
        train_nontarget_data, _ = self.permute_data(train_nontarget_data)
        train_target_fold_data = [train_target_data[i*target_fold_num:(i+1)*target_fold_num]
                                  for i in range(n_fold)]
        train_nontarget_fold_data = [train_nontarget_data[i * target_fold_num:(i + 1) * target_fold_num]
                                     for i in range(n_fold)]
        if train_target_fold_data[-1].shape[0] < train_nontarget_fold_data[-1].shape[0]:
            train_nontarget_fold_data[-1] = train_nontarget_fold_data[-1][:train_target_fold_data[-1].shape[0]]
        train_target_fold_label = [np.ones(train_target_fold_data[i].shape[0]) for i in range(n_fold)]
        train_nontarget_fold_label = [np.zeros(train_nontarget_fold_data[i].shape[0]) for i in range(n_fold)]
        train_val_x_list = [
            np.concatenate((train_target_fold_data[i], train_nontarget_fold_data[i]), axis=0)
            for i in range(n_fold)]
        train_val_y_list = [
            np.concatenate((train_target_fold_label[i], train_nontarget_fold_label[i]), axis=0)
            for i in range(n_fold)]
        for i in range(n_fold):
            train_val_x_list[i], train_val_y_list[i] = self.permute_data(train_val_x_list[i], train_val_y_list[i])

        return [train_val_x_list, train_val_y_list], [test_set_x, test_set_y]

    def split_train_validation_test_1block(self, x_data, y_data, n_fold):
        # target_label_index = np.where(y_data == 1)[0]
        # nontarget_label_index = np.where(y_data == 2)[0]
        # target_data = x_data[target_label_index]
        # nontarget_data = x_data[nontarget_label_index]
        # total_target_num = len(target_label_index)
        # total_nontarget_num = len(nontarget_label_index)
        # target_num_for_train = np.ceil(total_target_num * 0.8).astype(int)
        # nontarget_num_for_train = np.ceil(total_nontarget_num * 0.8).astype(int)
        # train_data = np.concatenate([target_data[:target_num_for_train], nontarget_data[:nontarget_num_for_train]], axis=0)
        # train_labels = np.concatenate([np.ones(target_num_for_train), np.zeros(nontarget_num_for_train)], axis=0)
        # test_data = np.concatenate([target_data[target_num_for_train:], nontarget_data[nontarget_num_for_train:]], axis=0)
        # test_labels = np.concatenate([np.ones(total_target_num - target_num_for_train),
        #                               np.zeros(total_nontarget_num - nontarget_num_for_train)], axis=0)
        num_for_train = np.ceil(x_data.shape[0] * 0.8).astype(int)
        train_data, train_labels = x_data[:num_for_train], y_data[:num_for_train]
        test_data, test_labels = x_data[num_for_train:], y_data[num_for_train:]
        train_data, train_labels = self.permute_data(train_data, train_labels)
        nontarget_index_test = np.where(test_labels == 2)[0]
        test_labels[nontarget_index_test] = 0
        # 将train_data进一步划分成训练集和验证集，共n份，对每一类数据单独划分
        target_num_index = np.where(train_labels == 1)[0]
        nontarget_num_index = np.where(train_labels == 2)[0]
        target_fold_num = len(target_num_index) // n_fold + 1
        nontarget_fold_num = len(nontarget_num_index) // n_fold + 1
        train_target_data = train_data[target_num_index]
        train_nontarget_data = train_data[nontarget_num_index]
        # 打乱训练集
        train_target_data, _ = self.permute_data(train_target_data)
        train_nontarget_data, _ = self.permute_data(train_nontarget_data)
        train_target_fold_data = [train_target_data[i * target_fold_num:(i + 1) * target_fold_num]
                                    for i in range(n_fold)]
        train_nontarget_fold_data = [train_nontarget_data[i * nontarget_fold_num:(i + 1) * nontarget_fold_num]
                                        for i in range(n_fold)]
        train_target_fold_label = [np.ones(train_target_fold_data[i].shape[0]) for i in range(n_fold)]
        train_nontarget_fold_label = [np.zeros(train_nontarget_fold_data[i].shape[0]) for i in range(n_fold)]
        train_val_x_list = [
            np.concatenate((train_target_fold_data[i], train_nontarget_fold_data[i]), axis=0)
            for i in range(n_fold)]
        train_val_y_list = [
            np.concatenate((train_target_fold_label[i], train_nontarget_fold_label[i]), axis=0)
            for i in range(n_fold)]
        for i in range(n_fold):
            train_val_x_list[i], train_val_y_list[i] = self.permute_data(train_val_x_list[i], train_val_y_list[i])

        return [train_val_x_list, train_val_y_list], [test_data, test_labels]


def load_blocks(data, num_blocks=3):
    """加载处理CAS数据集"""

    blocks = [data[0, i] for i in range(num_blocks)]
    data_blocks = [block[:-1, :] for block in blocks]
    trigger_channel = [block[-1, :] for block in blocks]
    trigger_positions = [np.where(trigger != 0)[0] for trigger in trigger_channel]
    class_labels = [trigger[trigger != 0] for trigger in trigger_channel]
    return data_blocks, trigger_positions, class_labels


def load_data(path):
    """根据文件提取出数据集的数据 触发位置 标签信息"""

    # 初始化变量
    sub_data, trigger_positions, class_labels = None, None, None
    fs = None

    # 从mat文件中提取数据，数据已存到DataSet数据集
    dataset_name = os.path.dirname(path).split(os.path.sep)[2]
    total_data = sio.loadmat(path)

    if dataset_name == "THU":
        fs = 250
        data_block1, data_block2 = total_data["EEGdata1"], total_data["EEGdata2"]
        # 将数据类型更改为 float32
        data_block1 = data_block1.astype(np.float32)
        data_block2 = data_block2.astype(np.float32)
        sub_data = [data_block1, data_block2]
        trigger_positions = [total_data["trigger_positions"][0], total_data["trigger_positions"][1]]
        class_labels = [total_data["class_labels"][0], total_data["class_labels"][1]]
    elif dataset_name == "CAS":
        fs = 1000
        subA_data, subA_trigger_positions, subA_class_labels = load_blocks(total_data["Sa"])  # [3]
        subB_data, subB_trigger_positions, subB_class_labels = load_blocks(total_data["Sb"])
        # 整理两个受试者的数据
        subA_data = [block.astype(np.float32) for block in subA_data]
        subB_data = [block.astype(np.float32) for block in subB_data]
        sub_data = subA_data + subB_data
        trigger_positions = subA_trigger_positions + subB_trigger_positions
        class_labels = subA_class_labels + subB_class_labels
    elif dataset_name == "GIST":
        fs = 512
        data_block = total_data["data"]
        sub_data = [data_block.astype(np.float32)]
        class_labels = [total_data["label"][0][:600]]
        trigger_positions = [total_data["trigger_position"][0][:600]]

    # 确保变量在返回之前已经被赋值
    if sub_data is None or trigger_positions is None or class_labels is None:
        # 这里可以根据实际情况处理，比如抛出异常，或者返回空列表/None等
        raise ValueError("The dataset could not be loaded. Please check the dataset name and path.")

    return sub_data, trigger_positions, class_labels, fs


def save_data(sub_data, class_labels, folder_path, sub_name):
    # 将预处理后的数据保存到对应文件夹中
    try:
        # 创建文件夹路径
        os.makedirs(folder_path, exist_ok=True)

        # 保存数据到MAT文件中
        sio.savemat(os.path.join(folder_path, f"{sub_name}.mat"),
                    {"x_data": sub_data, "y_data": class_labels})
        print(f"Data saved successfully to {folder_path}")
    except Exception as e:
        print(f"Error occurred while saving data: {e}")


class ChannelLocator:
    def __init__(self, loc_file_path):
        self.loc_file_path = loc_file_path
        self.channel_regions = {}
        self.channel_index_to_name = {}

    def load_loc_file(self):
        with open(self.loc_file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                channel_name = parts[-1]
                self.channel_index_to_name[channel_name] = int(parts[0]) - 1

    def create_channel_regions(self):
        region_names = ["Pre-frontal", "Left temporal", "Frontal", "Right temporal",
                        "Left parietal", "Central", "Right parietal", "Occipital"]
        # region_indices = [['FP1', 'FPz', 'FP2', 'AF3', 'AF4'],
        #                   ['F7', 'F5', 'F3', 'FT7', 'FC5', 'FC3', 'T7', 'C5', 'C3'],
        #                   ['F1', 'Fz', 'F2', 'FC1', 'FCz', 'FC2'],
        #                   ['F8', 'F6', 'F4', 'FT8', 'FC6', 'FC4', 'T8', 'C6', 'C4'],
        #                   ['TP7', 'CP5', 'CP3', 'P7', 'P5', 'P3'],
        #                   ['C1', 'Cz', 'C2', 'CP1', 'CPz', 'CP2', 'P1', 'Pz', 'P2'],
        #                   ['TP8', 'CP6', 'CP4', 'P8', 'P6', 'P4'],
        #                   ['PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO7', 'PO8', 'CB1', 'CB2', 'O1', 'Oz', 'O2']]
        region_indices = [['FP1', 'FPz', 'FP2', 'AF3', 'AF4'],
                          ['F7', 'F5', 'F3', 'FT7', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8'],
                          ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'C2', 'C4', 'C6'],
                          ['FT7', 'T7', 'TP7'],
                          ['FT8', 'T8', 'TP8'],
                          ['CP5', 'Cp3', 'CP1', 'P7', 'P5', 'P3', 'P1'],
                          ['CP6', 'CP4', 'CP2', 'P8', 'P6', 'P4', 'P2'],
                          ['PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO7', 'PO8', 'CB1', 'CB2', 'O1', 'Oz', 'O2']]

        channels_processed = [[self.channel_index_to_name.get(channel, None) for channel in region]
                              for region in region_indices]
        channels_processed = [[int(channel) for channel in region if channel is not None]
                              for region in channels_processed]
        self.channel_regions = dict(zip(region_names, channels_processed))

    def process_loc_file(self):
        self.load_loc_file()
        self.create_channel_regions()

    def get_channel_regions(self):
        return self.channel_regions

    def get_channel_dictionary(self):
        return self.channel_index_to_name

    def get_channel_regions_index(self):
        region_list = []
        for value in self.channel_regions.values():
            region_list.append(value)
        return region_list


if __name__ == '__main__':
    loc_file_path = r"..\Dataset\THU\64-channels_calibration.loc"
    locator = ChannelLocator(loc_file_path)
    locator.process_loc_file()

    channel_regions = locator.get_channel_regions()
    channel_dictionary = locator.get_channel_dictionary()
    print("Channel regions:")
    print(channel_regions)
    print("\nChannel dictionary:")
    print(channel_dictionary)
    channel_regions_list = locator.get_channel_regions_index()
    print("\nChannel regions list:")
    print(channel_regions_list)
