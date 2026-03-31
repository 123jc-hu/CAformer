import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
from Data_Processing.data_process import TrainTestSplit
import os
import argparse
import numpy as np


dataset_dict = {
    "THU": r"D:\learning_softwares\Collaborative_RSVP\Dataset\THU\preprocessed",
    "CAS": r".\Dataset\CAS\preprocessed",
    "GIST": r".\Dataset\GIST\preprocessed",
}


# 划分训练、验证、测试集
def load_preprocessed_data(args):
    data_path = os.path.join(dataset_dict[args.dataset], args.sub_name)
    remove_num = args.remove_num
    n_fold = args.n_fold
    # 读取预处理后的数据
    all_data = sio.loadmat(data_path)
    x_data, y_data = all_data["x_data"], all_data["y_data"]  # 这里的x_data和y_data都不是list
    train_val_test_module = TrainTestSplit()

    # 去除目标前后n个非目标
    x_data_list, y_data_list = [], []
    block_num = len(x_data)
    for i in range(block_num):
        x_data_remove, y_data_remove = train_val_test_module.nontarget_data_remove(remove_num, x_data[i], y_data[i])
        # 打乱数据
        x_data_remove, y_data_remove = train_val_test_module.permute_data(x_data_remove, y_data_remove)
        x_data_list.append(x_data_remove)
        y_data_list.append(y_data_remove)

    # 划分数据集 [x_train, y_train]
    train_val_Set, test_Set = train_val_test_module.split_train_validation_test(x_data_list, y_data_list, n_fold)

    return train_val_Set, test_Set


def load_preprocessed_data_2days(args):
    remove_num = args.remove_num
    n_fold = args.n_fold
    train_val_test_module = TrainTestSplit()

    sub_dict = ["A", "B"]
    x_data_list, y_data_list = [], []
    for i in range(2):
        sub_name = args.sub_name + sub_dict[i]
        data_path = os.path.join(dataset_dict[args.dataset], sub_name)
        # 读取预处理后的数据
        all_data = sio.loadmat(data_path)
        x_data, y_data = all_data["x_data"], all_data["y_data"]  # 这里的x_data和y_data都不是list

        # 去除目标前后n个非目标
        block_num = len(x_data)
        for j in range(block_num):
            x_data_remove, y_data_remove = train_val_test_module.nontarget_data_remove(remove_num, x_data[j], y_data[j])
            # 打乱数据
            x_data_remove, y_data_remove = train_val_test_module.permute_data(x_data_remove, y_data_remove)
            x_data_list.append(x_data_remove)
            y_data_list.append(y_data_remove)
    # 组合两天的每个block
    block_num = len(x_data_list) // 2
    for index in range(block_num):
        x_data_list[index] = np.concatenate((x_data_list[index], x_data_list[index+block_num]), axis=0)
        y_data_list[index] = np.concatenate((y_data_list[index], y_data_list[index + block_num]), axis=0)
    # 划分数据集 [x_train, y_train]
    train_val_Set, test_Set = train_val_test_module.split_train_validation_test(
        x_data_list[:block_num], y_data_list[:block_num], n_fold)

    return train_val_Set, test_Set

def load_preprocessed_data_2days_like_mtcn(args):
    remove_num = args.remove_num
    n_fold = args.n_fold
    train_val_test_module = TrainTestSplit()

    sub_dict = ["A", "B"]
    x_data_list, y_data_list = [], []
    for i in range(2):
        sub_name = args.sub_name + sub_dict[i]
        data_path = os.path.join(dataset_dict[args.dataset], sub_name)
        # 读取预处理后的数据
        all_data = sio.loadmat(data_path)
        x_data, y_data = all_data["x_data"], all_data["y_data"]
        # 去除目标前后n个非目标
        block_num = len(x_data)
        for j in range(block_num):
            x_data_remove, y_data_remove = train_val_test_module.nontarget_data_remove(remove_num, x_data[j], y_data[j])
            # 打乱数据
            x_data_remove, y_data_remove = train_val_test_module.permute_data(x_data_remove, y_data_remove)
            x_data_list.append(x_data_remove)
            y_data_list.append(y_data_remove)
    # 划分数据集 [x_train, y_train]
    block_num = len(x_data_list) // 2
    x_train = np.concatenate(x_data_list[:block_num])
    y_train = np.concatenate(y_data_list[:block_num])
    x_test = np.concatenate(x_data_list[block_num:])
    y_test = np.concatenate(y_data_list[block_num:])
    train_val_Set, test_Set = train_val_test_module.split_train_validation_test([x_train, x_test], [y_train, y_test], n_fold)
    return train_val_Set, test_Set

def load_preprocessed_data_for_GIST(args):
    remove_num = args.remove_num
    n_fold = args.n_fold
    train_val_test_module = TrainTestSplit()

    data_path = os.path.join(dataset_dict[args.dataset], args.sub_name)
    # 读取预处理后的数据
    all_data = sio.loadmat(data_path)
    x_data, y_data = all_data["x_data"][0], all_data["y_data"][0]  # 这里的x_data和y_data都不是list
    # # 去除目标前后n个非目标
    # x_data_remove, y_data_remove = train_val_test_module.nontarget_data_remove(remove_num, x_data[0], y_data[0])
    # 打乱数据
    # x_data_remove, y_data_remove = train_val_test_module.permute_data(x_data, y_data)
    # 划分数据集 [x_train, y_train]
    train_val_Set, test_Set = train_val_test_module.split_train_validation_test_1block(x_data, y_data, n_fold)

    return train_val_Set, test_Set


def data_rebalanced(data, label):
    # data -> list data[i] -> array
    extra_nontarget_data = []
    for i in range(len(data)):
        target_index = np.where(label[i] == 1)[0]
        nontarget_index = np.where(label[i] == 0)[0]
        permutation = np.random.permutation(len(nontarget_index))  # 打乱顺序再提取和target相同数量的样本
        new_nontarget_index = nontarget_index[permutation[:len(target_index)]]
        extra_nontarget_data.append(nontarget_index[permutation[len(target_index):]])
        new_index = np.concatenate([target_index, new_nontarget_index])
        data[i] = data[i][new_index]
        label[i] = label[i][new_index]
    return [data, label], np.concatenate(extra_nontarget_data)


class SupportQuerySplit:
    """划分支持集和查询集"""
    def __init__(self,
                 x_data,
                 y_data,
                 episodes_per_epoch: int = None,
                 ns: int = None,
                 nq: int = None):
        """
        传入训练集和支持集查询集数量Ns Nq，生成episodes_per_epoch个数据并打包成dataloader
        :param x_data: 训练集数据 (N, C, T)
        :param y_data: 训练集标签
        :param episodes_per_epoch: 每个epoch需要进行的迭代次数，相当于num_batch
        :param ns: 支持集每类样本数量
        :param nq: 查询集每类样本数量
        """
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.episodes_per_epoch = episodes_per_epoch
        self.Ns = ns
        self.Nq = nq
        self.train_val_split = TrainTestSplit()

    def support_query_set_split(self, mode="train"):
        # 给定数据集，从里面划分出支持集和查询集，按50%来划分，不然验证集里面查询集的目标太少了
        # 直接输出两类数据
        target_data = self.x_data[np.where(self.y_data == 1)[0]]
        nontarget_data = self.x_data[np.where(self.y_data == 0)[0]]
        target_data, _ = self.train_val_split.permute_data(target_data)
        nontarget_data, _ = self.train_val_split.permute_data(nontarget_data)

        self.Nq = target_data.shape[0] - self.Ns if mode == "validate" and self.Nq is None else self.Nq
        batch_size = (self.Ns+self.Nq)*2

        batch_x, batch_y = [], []
        for _ in range(self.episodes_per_epoch):
            target_data_sampled = self.random_sample_data(target_data, self.Ns+self.Nq)
            if mode == "validate":
                nontarget_data_sampled = self.random_sample_data(nontarget_data, nontarget_data.shape[0])
            else:
                nontarget_data_sampled = self.random_sample_data(nontarget_data, self.Ns + self.Nq)
            support_set_x = np.concatenate((nontarget_data_sampled[:self.Ns], target_data_sampled[:self.Ns]))
            support_set_y = np.concatenate((np.zeros(self.Ns), np.ones(self.Ns)))
            query_set_x = np.concatenate((nontarget_data_sampled[self.Ns:], target_data_sampled[self.Ns:]))
            if mode == "validate":
                # query_set_x = np.concatenate((nontarget_data_sampled[self.Ns:], target_data_sampled[self.Ns:]))
                query_set_y = np.concatenate((np.zeros(nontarget_data_sampled[self.Ns:].shape[0]), np.ones(self.Nq)))
            else:
                query_set_y = np.concatenate((np.zeros(self.Nq), np.ones(self.Nq)))
            minibatch_x = np.concatenate((support_set_x, query_set_x))  # 为了方便直接用2*Ns划分支持集和查询集
            minibatch_y = np.concatenate((support_set_y, query_set_y))
            batch_x.append(minibatch_x)
            batch_y.append(minibatch_y)
        batch_size = batch_x[0].shape[0]
        return np.concatenate(batch_x), np.concatenate(batch_y), batch_size

    def random_sample_data(self, data, n):
        N = data.shape[0]  # 总样本数量
        assert n <= N, "采样数量不能大于总样本数量"
        # 生成随机排列的索引
        indices = np.random.choice(N, n, replace=False)

        # 使用索引从原始数据中选择样本
        sampled_data = data[indices]
        return sampled_data


class SupportQuerySplitSeparately:
    """划分支持集和查询集"""
    def __init__(self,
                 x_data,
                 y_data,
                 episodes_per_epoch: int = None,
                 ns0: int = None,
                 ns1: int = None,
                 nq: int = None):
        """
        传入训练集和支持集查询集数量Ns Nq，生成episodes_per_epoch个数据并打包成dataloader
        :param x_data: 训练集数据 (N, C, T)
        :param y_data: 训练集标签
        :param episodes_per_epoch: 每个epoch需要进行的迭代次数，相当于num_batch
        :param ns0: 支持集每类样本数量
        :param nq: 查询集每类样本数量
        """
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.episodes_per_epoch = episodes_per_epoch
        self.Ns0 = ns0
        self.Ns1 = ns1
        self.Nq = nq
        self.train_val_split = TrainTestSplit()

    def support_query_set_split(self, mode="train"):
        # 给定数据集，从里面划分出支持集和查询集，按50%来划分，不然验证集里面查询集的目标太少了
        # 直接输出两类数据
        target_data = self.x_data[np.where(self.y_data == 1)[0]]
        nontarget_data = self.x_data[np.where(self.y_data == 0)[0]]
        target_data, _ = self.train_val_split.permute_data(target_data)
        nontarget_data, _ = self.train_val_split.permute_data(nontarget_data)

        self.Nq = target_data.shape[0] - self.Ns1 if mode == "validate" and self.Nq is None else self.Nq
        batch_size = (self.Ns1+self.Nq) + (self.Ns0+self.Nq)

        batch_x, batch_y = [], []
        for _ in range(self.episodes_per_epoch):
            target_data_sampled = self.random_sample_data(target_data, self.Ns1+self.Nq)
            if mode == "validate":
                nontarget_data_sampled = self.random_sample_data(nontarget_data, nontarget_data.shape[0])
            else:
                nontarget_data_sampled = self.random_sample_data(nontarget_data, self.Ns0 + self.Nq)
            support_set_x = np.concatenate((nontarget_data_sampled[:self.Ns0], target_data_sampled[:self.Ns1]))
            support_set_y = np.concatenate((np.zeros(self.Ns0), np.ones(self.Ns1)))
            query_set_x = np.concatenate((nontarget_data_sampled[self.Ns0:], target_data_sampled[self.Ns1:]))
            if mode == "validate":
                # query_set_x = np.concatenate((nontarget_data_sampled[self.Ns:], target_data_sampled[self.Ns:]))
                query_set_y = np.concatenate((np.zeros(nontarget_data_sampled[self.Ns0:].shape[0]), np.ones(self.Nq)))
            else:
                query_set_y = np.concatenate((np.zeros(self.Nq), np.ones(self.Nq)))
            minibatch_x = np.concatenate((support_set_x, query_set_x))  # 为了方便直接用2*Ns划分支持集和查询集
            minibatch_y = np.concatenate((support_set_y, query_set_y))
            batch_x.append(minibatch_x)
            batch_y.append(minibatch_y)
        # batch_size = batch_x[0].shape[0]
        return np.concatenate(batch_x), np.concatenate(batch_y), batch_size

    def random_sample_data(self, data, n):
        N = data.shape[0]  # 总样本数量
        assert n <= N, "采样数量不能大于总样本数量"
        # 生成随机排列的索引
        indices = np.random.choice(N, n, replace=False)

        # 使用索引从原始数据中选择样本
        sampled_data = data[indices]
        return sampled_data


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x = x_data
        self.y = y_data
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.x)


def dataloader(x_data, y_data, transform=None, batch_size=128, shuffle=True, num_workers=0, pin_memory=False):
    torch.manual_seed(2024)
    # 将np.array数据转换为tensor
    x_data = torch.as_tensor(np.array(x_data), dtype=torch.float32)
    y_data = torch.as_tensor(np.array(y_data), dtype=torch.int32)
    # 如果x_data维度是3，则增加一个维度(N, C, T)->(N, 1, C, T)
    if len(x_data.shape) == 3:
        x_data = x_data.unsqueeze(dim=1)
    # dataset = CustomDataset(x_data, y_data, transform=transform)
    dataset = TensorDataset(x_data, y_data)
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


def dataloader_with_aug(x_data, y_data, x_aug1, x_aug2, transform=None, batch_size=128, shuffle=True, num_workers=0, pin_memory=False):
    torch.manual_seed(2024)
    # 将np.array数据转换为tensor
    x_data = torch.as_tensor(np.array(x_data), dtype=torch.float32)
    y_data = torch.as_tensor(np.array(y_data), dtype=torch.int32)
    x_aug1 = torch.as_tensor(np.array(x_aug1), dtype=torch.float32)
    x_aug2 = torch.as_tensor(np.array(x_aug2), dtype=torch.float32)
    # 如果x_data维度是3，则增加一个维度(N, C, T)->(N, 1, C, T)
    if len(x_data.shape) == 3:
        x_data = x_data.unsqueeze(dim=1)
        x_aug1 = x_aug1.unsqueeze(dim=1)
        x_aug2 = x_aug2.unsqueeze(dim=1)
    # dataset = CustomDataset(x_data, y_data, transform=transform)
    dataset = TensorDataset(x_data, y_data, x_aug1, x_aug2)
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)



class MTCNDataset(Dataset):
    def __init__(self, x_data, y_data, x_mtr, y_mtr, x_msr, y_msr):
        self.x_data = torch.as_tensor(np.array(x_data), dtype=torch.float32)
        self.y_data = torch.as_tensor(np.array(y_data), dtype=torch.int32)
        self.x_mtr = torch.as_tensor(np.array(x_mtr), dtype=torch.float32)
        self.y_mtr = torch.as_tensor(np.array(y_mtr), dtype=torch.int32)
        self.x_msr = torch.as_tensor(np.array(x_msr), dtype=torch.float32)
        self.y_msr = torch.as_tensor(np.array(y_msr), dtype=torch.int32)

        # Add an additional dimension if the data is 3D
        if len(self.x_data.shape) == 3:
            self.x_data = self.x_data.unsqueeze(dim=1)
            self.x_mtr = self.x_mtr.unsqueeze(dim=1)
            self.x_msr = self.x_msr.unsqueeze(dim=1)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = self.x_data[index, ...]
        y = self.y_data[index, ...]
        x_mtr = self.x_mtr[9 * index: 9 * index + 9, ...]
        y_mtr = self.y_mtr[9 * index: 9 * index + 9, ...]
        x_msr = self.x_msr[8 * index: 8 * index + 8, ...]
        y_msr = self.y_msr[8 * index: 8 * index + 8, ...]
        return x, y, x_mtr, y_mtr, x_msr, y_msr


def dataloader_for_MTCN(x_data, y_data, x_mtr, y_mtr, x_msr, y_msr, batch_size=128, shuffle=True, num_workers=0,
                        pin_memory=False):
    torch.manual_seed(2024)
    dataset = MTCNDataset(x_data, y_data, x_mtr, y_mtr, x_msr, y_msr)
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Learning Models")
    parser.add_argument("--dataset", type=str, default="THU", help="choose dataset name")
    parser.add_argument("--sub_name", type=str, default="sub1A", help="choose sub for each dataset")
    parser.add_argument("--remove_num", type=int, default=1, help="number of removed non-targets")
    parser.add_argument("--n_fold", type=int, default=5, help="N fold cross-validation")
    args = parser.parse_args()
    train_val_set, test_set = load_preprocessed_data(args)
    data_dict = {}
    data_dict["fold_data"] = [torch.tensor(train_data) for train_data in train_val_set[0]]
    data_dict["fold_label"] = [torch.tensor(train_label) for train_label in train_val_set[1]]
    data_dict["X_test"] = torch.tensor(test_set[0])
    data_dict["Y_test"] = torch.tensor(test_set[1])

    torch.save(data_dict, "..\\Dataset\\THU\\preprocessed\\sub1.pth")
