import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics


class EarlyStopping:
    """Early stop training when validation loss stops improving."""

    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
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
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def cal_auc(y_true, y_score):
    return metrics.roc_auc_score(y_true, y_score)


def cal_ba(y_true, y_pred):
    return metrics.balanced_accuracy_score(y_true, y_pred)


def cal_F1_score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred)


def cal_confusion_matrix(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred)


def calculate_tpr_tnr(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return tpr, tnr


def plot_train_val_figure(train_loss_list, val_loss_list, val_auc_list, val_ba_list, path_folder, fold_i, sub_name):
    x = np.arange(len(train_loss_list))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, np.array(train_loss_list), "b", label="Train Loss")
    ax.plot(x, np.array(val_loss_list), "g", label="Val Loss")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.savefig(os.path.join(path_folder, f"{sub_name}_fold{fold_i}_Loss_figure"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, np.array(val_auc_list), "r", label="Val AUC")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    fig.savefig(os.path.join(path_folder, f"{sub_name}_fold{fold_i}_AUC_figure"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, np.array(val_ba_list), "r", label="Val BA")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BA")
    fig.savefig(os.path.join(path_folder, f"{sub_name}_fold{fold_i}_BA_figure"))
    plt.close(fig)
