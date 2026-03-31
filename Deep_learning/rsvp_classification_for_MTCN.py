import torch
import torch.nn as nn
import numpy as np
import os
from Deep_learning.exp_basic import ExpBasic
from Data_Processing.make_dataset import dataloader, dataloader_for_MTCN
from Deep_learning.tools import EarlyStopping, cal_auc, cal_ba, cal_F1_score, cal_confusion_matrix, \
    calculate_tpr_tnr, calculate_kappa
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import random
import matplotlib.pyplot as plt
from pathlib import Path


class RsvpClassification_for_MTCN(ExpBasic):
    def __init__(self, args):
        super(RsvpClassification_for_MTCN, self).__init__(args)

    def _build_model(self):
        return self.model_dict[self.args.model].Model(self.args).float()

    def _select_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    def _select_criterion(self, y_data):
        target_num = len(np.where(y_data == 1)[0])
        nontarget_num = len(np.where(y_data == 0)[0])
        class_weight = torch.tensor([1, nontarget_num / target_num]).to(self.device)
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
        total_outputs, total_label = [], []
        total_loss = []
        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(vali_loader):
                x_val_main_task, y_val_main_task, x_val_vto_task, y_val_vto_task, x_val_msp_task, y_val_msp_task = data
                x_val_main_task = x_val_main_task.type(torch.FloatTensor)
                x_val_vto_task = x_val_vto_task.type(torch.FloatTensor)
                x_val_msp_task = x_val_msp_task.type(torch.FloatTensor)

                x_val_main_task, y_val_main_task = x_val_main_task.to(self.device), y_val_main_task.to(self.device)
                x_val_vto_task, y_val_vto_task = x_val_vto_task.to(self.device), y_val_vto_task.to(self.device)
                x_val_msp_task, y_val_msp_task = x_val_msp_task.to(self.device), y_val_msp_task.to(self.device)

                x_val_vto_task = x_val_vto_task.reshape(x_val_vto_task.shape[0] * x_val_vto_task.shape[1],
                                                        x_val_vto_task.shape[2], x_val_vto_task.shape[3], x_val_vto_task.shape[4])
                x_val_msp_task = x_val_msp_task.reshape(x_val_msp_task.shape[0] * x_val_msp_task.shape[1],
                                                        x_val_msp_task.shape[2], x_val_msp_task.shape[3], x_val_msp_task.shape[4])
                y_val_vto_task = y_val_vto_task.reshape(y_val_vto_task.shape[0] * y_val_vto_task.shape[1])
                y_val_msp_task = y_val_msp_task.reshape(y_val_msp_task.shape[0] * y_val_msp_task.shape[1])

                pred_primary, loss_main = self.model(x_val_main_task, "main")
                pred_vto, loss_vto = self.model(x_val_vto_task, "vto")
                pred_msp, loss_msp = self.model(x_val_msp_task, "msp")

                loss = criterion.calculateTrainStageLoss(pred_primary, y_val_main_task, pred_vto, y_val_vto_task,
                                                         pred_msp, y_val_msp_task)
                loss += loss_main + loss_vto + loss_msp

                total_label.append(y_val_main_task)
                preds.append(pred_primary)
                total_loss.append(loss)

            total_label = torch.cat(total_label, 0)
            preds = torch.cat(preds, 0)
            total_loss = torch.tensor(total_loss).mean()

        predictions = torch.argmax(preds, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = total_label.cpu().numpy()
        auc = cal_auc(trues, preds[:, 1].cpu().numpy())
        ba = cal_ba(trues, predictions)

        return total_loss.cpu().numpy(), auc, ba

    def train(self, setting, fold_i, train_val_set, train_val_mtr_set=None, train_val_msr_set=None, verbose=True):
        train_val_set_here = copy.deepcopy(train_val_set)
        train_val_mtr_set_here = copy.deepcopy(train_val_mtr_set)
        train_val_msr_set_here = copy.deepcopy(train_val_msr_set)
        val_x, val_y = train_val_set_here[0].pop(fold_i), train_val_set_here[1].pop(fold_i)
        val_x_mtr, val_y_mtr = train_val_mtr_set_here[0].pop(fold_i), train_val_mtr_set_here[1].pop(fold_i)
        val_x_msr, val_y_msr = train_val_msr_set_here[0].pop(fold_i), train_val_msr_set_here[1].pop(fold_i)
        train_x = np.concatenate(train_val_set_here[0], axis=0)
        train_y = np.concatenate(train_val_set_here[1], axis=0)
        train_x_mtr = np.concatenate(train_val_mtr_set_here[0], axis=0)
        train_y_mtr = np.concatenate(train_val_mtr_set_here[1], axis=0)
        train_x_msr = np.concatenate(train_val_msr_set_here[0], axis=0)
        train_y_msr = np.concatenate(train_val_msr_set_here[1], axis=0)
        train_loader = dataloader_for_MTCN(train_x, train_y, train_x_mtr, train_y_mtr, train_x_msr, train_y_msr,
                                           batch_size=self.args.batch_size)
        val_loader = dataloader_for_MTCN(val_x, val_y, val_x_mtr, val_y_mtr, val_x_msr, val_y_msr,
                                         batch_size=self.args.batch_size)

        path = os.path.join('.\\checkpoints', setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,
                                       path=os.path.join(path, f'{self.args.sub_name}_fold{fold_i+1}_checkpoint.pth'))

        model_optim = self._select_optimizer()
        criterion = LossFunction()
        scheduler = ReduceLROnPlateau(model_optim, 'min', factor=0.5, patience=5, verbose=True)

        train_loss_list, val_loss_list, val_auc_list, val_ba_list = [], [], [], []
        for epoch in range(self.args.train_epochs):
            current_lr = model_optim.param_groups[0]["lr"]
            train_loss = []

            self.model.train()
            for i, data in enumerate(train_loader):
                model_optim.zero_grad()
                x_train_main_task, y_train_main_task, x_train_vto_task, y_train_vto_task, x_train_msp_task, y_train_msp_task = data
                if x_train_main_task.shape[0] < 10:
                    continue
                x_train_main_task = x_train_main_task.type(torch.FloatTensor)
                x_train_vto_task = x_train_vto_task.type(torch.FloatTensor)
                x_train_msp_task = x_train_msp_task.type(torch.FloatTensor)
                x_train_main_task, y_train_main_task = x_train_main_task.to(self.device), y_train_main_task.to(
                    self.device)
                x_train_vto_task, y_train_vto_task = x_train_vto_task.to(self.device), y_train_vto_task.to(self.device)
                x_train_msp_task, y_train_msp_task = x_train_msp_task.to(self.device), y_train_msp_task.to(self.device)

                x_train_vto_task = x_train_vto_task.reshape(x_train_vto_task.shape[0] * x_train_vto_task.shape[1],
                                                            x_train_vto_task.shape[2], x_train_vto_task.shape[3], x_train_vto_task.shape[4])
                x_train_msp_task = x_train_msp_task.reshape(x_train_msp_task.shape[0] * x_train_msp_task.shape[1],
                                                            x_train_msp_task.shape[2], x_train_msp_task.shape[3], x_train_msp_task.shape[4])
                y_train_vto_task = y_train_vto_task.reshape(y_train_vto_task.shape[0] * y_train_vto_task.shape[1])
                y_train_msp_task = y_train_msp_task.reshape(y_train_msp_task.shape[0] * y_train_msp_task.shape[1])

                pred_primary, loss_main = self.model(x_train_main_task, "main")
                pred_vto, loss_vto = self.model(x_train_vto_task, "vto")
                pred_msp, loss_msp = self.model(x_train_msp_task, "msp")

                loss = criterion.calculateTrainStageLoss(pred_primary, y_train_main_task, pred_vto,
                                                              y_train_vto_task,
                                                              pred_msp, y_train_msp_task)

                loss += loss_main + loss_vto + loss_msp
                train_loss.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            train_loss = np.mean(train_loss)
            val_loss, val_auc, val_ba = self.validate(val_loader, criterion)
            # save training information
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            val_auc_list.append(val_auc)
            val_ba_list.append(val_ba)

            print(
                f"{self.args.sub_name} Fold{fold_i + 1}/{self.args.n_fold} Epoch: {epoch + 1}, learning rate: "
                f"{current_lr} | Train Loss: {train_loss:.3f} Val Loss: {val_loss:.3f} Val AUC: {val_auc:.3f} "
                f"Val BA: {val_ba:.3f}")
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 调整学习率
            scheduler.step(val_loss)

        # 加载最佳模型
        self.model.load_state_dict(torch.load(early_stopping.path))

        # plot figure and save
        if verbose:
            epochs_range = list(range(1, len(train_loss_list) + 1))
            self.plot_and_save(
                epochs_range,
                [train_loss_list, val_loss_list],
                ["Train Loss", "Val Loss"],
                "Epochs",
                "Loss",
                os.path.join(path, f'{self.args.sub_name}_fold{fold_i + 1}_loss.png')
            )
            self.plot_and_save(
                epochs_range,
                [val_auc_list, val_ba_list],
                ["Val AUC", "Val BA"],
                "Epochs",
                "Score",
                os.path.join(path, f'{self.args.sub_name}_fold{fold_i + 1}_auc_ba.png')
            )

    def test(self, setting, fold_i, test_set, result_path):
        test_loader = dataloader(test_set[0], test_set[1], batch_size=self.args.batch_size, shuffle=False)
        print('Loading model...')
        checkpoint_path = Path('./checkpoints') / setting / f'{self.args.sub_name}_fold{fold_i + 1}_checkpoint.pth'

        self.model.load_state_dict(torch.load(checkpoint_path))
        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_x, label in test_loader:
                batch_x, label = batch_x.to(self.device), label.to(self.device)

                y_pred, _ = self.model(batch_x, "main")

                preds.append(y_pred)
                trues.append(label)

        preds, trues = torch.cat(preds), torch.cat(trues)

        predictions = torch.argmax(preds, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.cpu().numpy()

        auc = cal_auc(trues, preds[:, 1].cpu().numpy())
        ba = cal_ba(trues, predictions)
        f1_score = cal_F1_score(trues, predictions)
        conf_matrix = cal_confusion_matrix(trues, predictions)
        tpr, tnr = calculate_tpr_tnr(trues, predictions)
        kappa = calculate_kappa(trues, predictions)

        # result save
        # folder_path = os.path.join('./results/', setting)
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        result_str = (f'{self.args.sub_name} AUC:{auc:.4f} BA:{ba:.4f} '
                      f'conf_matrix:{conf_matrix} TPR:{tpr:.4f} TNR:{tnr:.4f}')
        print(result_str)

        with open(result_path, 'a') as f:
            f.write(f'{setting}\n')
            f.write(f'{result_str}\n\n')

        return auc.round(4), ba.round(4), f1_score.round(4), conf_matrix, tpr.round(4), tnr.round(4)


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


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.loss = AutomaticWeightedLoss(3)

    def calculateTrainStageLoss(self, pred_primary, label_primary, pred_vto, label_vto, pred_msp, label_msp):
        loss_primary = self.criterion(pred_primary, label_primary.type(torch.cuda.LongTensor))
        loss_vto = self.criterion(pred_vto, label_vto.type(torch.cuda.LongTensor))
        loss_msp = self.criterion(pred_msp, label_msp.type(torch.cuda.LongTensor))

        loss = self.loss(loss_primary, loss_vto, loss_msp)

        return loss

    def calculateTestStageILoss(self, pred_vto, label_vto, pred_msp, label_msp):
        loss_vto = self.criterion(pred_vto, label_vto.type(torch.cuda.LongTensor))
        loss_msp = self.criterion(pred_msp, label_msp.type(torch.cuda.LongTensor))

        loss = self.loss(loss_vto, loss_msp)

        return loss

    def calculateTestStageIILoss(self, pred_primary, label_primary):
        loss_primary = self.criterion(pred_primary, label_primary.type(torch.cuda.LongTensor))

        return loss_primary


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        length = len(x)-1
        for i, loss in enumerate(x):
            if i == length:
                loss_sum += 1 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
            else:
                loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
        return loss_sum

