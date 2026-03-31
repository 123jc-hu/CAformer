import torch
import torch.nn as nn
import numpy as np
import os
from Deep_learning.exp_basic import ExpBasic
from Data_Processing.make_dataset import dataloader
from Deep_learning.tools import EarlyStopping, cal_auc, cal_ba, cal_F1_score, cal_confusion_matrix, \
    plot_train_val_figure, calculate_tpr_tnr, calculate_kappa
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path


class RsvpClassification(ExpBasic):
    def __init__(self, args):
        super(RsvpClassification, self).__init__(args)

    def _build_model(self):
        return self.model_dict[self.args.model].Model(self.args).float()

    def _select_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    def _select_criterion(self, y_data):
        target_num = len(np.where(y_data == 1)[0])
        nontarget_num = len(np.where(y_data == 0)[0])
        class_weight = torch.tensor([1, (nontarget_num / target_num)]).to(self.device)
        return nn.CrossEntropyLoss(weight=class_weight)

    def plot_and_save(self, x, y_list, labels, xlabel, ylabel, filename):
        plt.figure(figsize=(10, 6))
        for y, label in zip(y_list, labels):
            plt.plot(x, y, label=label)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(filename)
        plt.clf()

    def validate(self, vali_loader, criterion):
        self.model.eval()


        return loss.cpu().numpy(), auc, ba

    def train(self, setting, fold_i, train_val_set, verbose=True):
        train_val_set_here = copy.deepcopy(train_val_set)
        val_x, val_y = train_val_set_here[0].pop(fold_i), train_val_set_here[1].pop(fold_i)
        train_x = np.concatenate(train_val_set_here[0], axis=0)
        train_y = np.concatenate(train_val_set_here[1], axis=0)
        train_loader = dataloader(train_x, train_y, batch_size=self.args.batch_size)
        val_loader = dataloader(val_x, val_y, batch_size=self.args.batch_size)

        path = os.path.join('.\\checkpoints', setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,
                                       path=os.path.join(path, f'{self.args.sub_name}_fold{fold_i+1}_checkpoint.pth'))

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(train_y)
        scheduler = ReduceLROnPlateau(model_optim, 'min', factor=0.5, patience=5, verbose=True)

        train_loss_list, val_loss_list, val_auc_list, val_ba_list = [], [], [], []
        for epoch in range(self.args.train_epochs):


    def test(self, setting, fold_i, test_set, result_path):

