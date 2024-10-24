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
        preds, total_outputs, total_label = [], [], []
        with torch.no_grad():
            for batch_x, label in vali_loader:
                batch_x, label = batch_x.to(self.device), label.to(self.device)
                outputs, _ = self.model(batch_x)
                total_outputs.append(outputs)
                total_label.append(label)
                preds.append(F.softmax(outputs, dim=-1))

            total_outputs = torch.cat(total_outputs)
            total_label = torch.cat(total_label)
            loss = criterion(total_outputs, total_label.long())

        preds = torch.cat(preds)
        predictions = torch.argmax(preds, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = total_label.cpu().numpy()
        auc = cal_auc(trues, preds[:, 1].cpu().numpy())
        ba = cal_ba(trues, predictions)

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
            current_lr = model_optim.param_groups[0]["lr"]
            train_loss = []

            self.model.train()
            for batch_x, label in train_loader:
                if batch_x.shape[0] < 10:
                    continue
                model_optim.zero_grad()
                batch_x, label = batch_x.to(self.device), label.to(self.device)
                outputs, _ = self.model(batch_x)
                loss = criterion(outputs, label.long())
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

                outputs, _ = self.model(batch_x)
                preds.append(F.softmax(outputs, dim=-1).detach())
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

        result_str = (f'{self.args.sub_name} AUC:{auc:.4f} BA:{ba:.4f} '
                      f'conf_matrix:{conf_matrix} TPR:{tpr:.4f} TNR:{tnr:.4f}')
        print(result_str)

        with open(result_path, 'a') as f:
            f.write(f'{setting}\n')
            f.write(f'{result_str}\n\n')

        return auc.round(4), ba.round(4), f1_score.round(4), conf_matrix, tpr.round(4), tnr.round(4)
