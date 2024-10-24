import os.path

import numpy as np
import torch
from sklearn import metrics
import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class EarlyStopping_with_C:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, C):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            torch.save(C, self.path.replace(".pth", "_C.pth"))
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class Gamma_reduce:
    def __init__(self, patience=7,):
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.gamma_reduce = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.gamma_reduce = True
        else:
            self.best_score = score
            self.counter = 0


def cal_auc(y_true, y_score):
    auc = metrics.roc_auc_score(y_true, y_score)
    return auc


def cal_ba(y_true, y_pred):
    ba = metrics.balanced_accuracy_score(y_true, y_pred)
    return ba


def cal_F1_score(y_true, y_pred):
    f1_score = metrics.f1_score(y_true, y_pred)
    return f1_score


def cal_confusion_matrix(y_ture, y_pred):
    # 计算混淆矩阵
    conf_matrix = metrics.confusion_matrix(y_ture, y_pred)
    return conf_matrix


def calculate_tpr_tnr(y_true, y_pred):
    # Compute the confusion matrix
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

    # Calculate TPR and FPR
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    tnr = tn / (tn + fp)

    return tpr, tnr


def calculate_kappa(y_true, y_pred):
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    return kappa


def plot_train_val_figure(train_loss_list, val_loss_list, val_auc_list, val_ba_list, path_folder, fold_i, sub_name):
    x = np.arange(len(train_loss_list))
    plt.figure(figsize=(10, 6))
    plt.plot(x, np.array(train_loss_list), "b", label="Train Loss")
    plt.plot(x, np.array(val_loss_list), "g", label="Val Loss")
    plt.legend()  # 显示图例
    plt.xlabel("Epoch")  # 设置横坐标轴标签
    plt.ylabel("Loss")  # 设置纵坐标轴标签
    plt.savefig(os.path.join(path_folder, f"{sub_name}_fold{fold_i}_Loss_figure"))
    plt.clf()

    # plt.figure(figsize=(12, 6))
    plt.plot(x, np.array(val_auc_list), "r", label="Val AUC")
    plt.legend()  # 显示图例
    plt.xlabel("Epoch")  # 设置横坐标轴标签
    plt.ylabel("AUC")  # 设置纵坐标轴标签
    plt.savefig(os.path.join(path_folder, f"{sub_name}_fold{fold_i}_AUC_figure"))
    plt.clf()

    # 绘制Balanced Accuracy曲线
    plt.plot(x, np.array(val_ba_list), "r", label="Val BA")
    plt.legend()  # 显示图例
    plt.xlabel("Epoch")  # 设置横坐标轴标签
    plt.ylabel("BA")  # 设置纵坐标轴标签
    plt.savefig(os.path.join(path_folder, f"{sub_name}_fold{fold_i}_BA_figure"))
    plt.close()  # 清除当前图形


def plot_EEG_data_figure(x_data, y_data, number=1, path=None):
    # x_data = (N, C, T)
    x_data, y_data = np.array(x_data), np.array(y_data)
    x = np.linspace(0, 1000, x_data.shape[-1])
    target_index = np.where(y_data == 1)[0]
    nontarget_data = np.where(y_data == 0)[0]
    for i in range(number):
        plt.figure(figsize=(10, 4))
        plt.xlabel("Time/s")
        plt.ylabel("Amplitude")
        plt.plot(x, x_data[target_index[i], 0, :], "r", label=f"target EEG number{i}")
        plt.plot(x, x_data[nontarget_data[i], 0, :], "b", label=f"non-target EEG number{i}")
        plt.legend()

    if path is not None:
        plt.savefig(path)
    plt.show()
    return None
