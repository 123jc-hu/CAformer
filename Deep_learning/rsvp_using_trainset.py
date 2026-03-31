import torch
import torch.nn as nn
import numpy as np
import os
from Deep_learning.exp_basic import ExpBasic
from Data_Processing.make_dataset import load_preprocessed_data, dataloader
from Deep_learning.tools import EarlyStopping, cal_auc, cal_ba, cal_F1_score, cal_confusion_matrix, \
    plot_train_val_figure
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt


class RsvpClassification_only_using_trainset(ExpBasic):
    def __init__(self, args):
        super(RsvpClassification_only_using_trainset, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _select_optimizer(self):
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                       betas=(self.args.beta1, self.args.beta2), weight_decay=self.args.weight_decay)
        return model_optim

    def _calculate_class_ratio(self, y_data):
        target_num = len(np.where(y_data == 1)[0])
        nontarget_num = len(np.where(y_data == 0)[0])
        class_weight = [1, (nontarget_num / target_num)]
        return class_weight

    def _select_criterion(self, class_weight):
        criterion = nn.CrossEntropyLoss(weight=class_weight.to(self.device))
        return criterion

    def train(self, setting, fold_i, train_val_set, verbose=True):
        # train_val_set, test_set = load_preprocessed_data(self.args)
        train_val_set_here = copy.deepcopy(train_val_set)
        train_x = np.concatenate(train_val_set_here[0], axis=0)
        train_y = np.concatenate(train_val_set_here[1], axis=0)
        train_loader = dataloader(train_x, train_y, batch_size=self.args.batch_size)
        class_weight = torch.tensor(self._calculate_class_ratio(train_y))

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,
                                       path=os.path.join(path, f'{self.args.sub_name}_fold{fold_i + 1}_checkpoint.pth'))

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(class_weight)
        scheduler = ReduceLROnPlateau(model_optim, 'min', factor=0.5, patience=5, verbose=True)

        train_loss_list = []
        for epoch in range(self.args.train_epochs):
            current_lr = model_optim.param_groups[0]["lr"]
            train_loss = []

            self.model.train()

            for i, (batch_x, label) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.to(self.device)
                label = label.to(self.device)
                target_label = torch.where(label == 1)[0]
                if len(target_label) == 0:
                    continue

                outputs, _ = self.model(batch_x)
                loss = criterion(outputs, label.long())
                train_loss.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            train_loss = torch.tensor(train_loss).mean()
            # save training information
            train_loss_list.append(train_loss)

            print(
                "{0} Fold{1}/{2} Epoch: {3}, learning rate: {4} | Train Loss: {5:.3f}"
                .format(self.args.sub_name, fold_i + 1, self.args.n_fold, epoch + 1, current_lr, train_loss))
            early_stopping(train_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 调整学习率
            scheduler.step(train_loss)

        # save best model
        best_model_path = os.path.join(path, f'{self.args.sub_name}_fold{fold_i + 1}_checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        # plot figure and save
        if verbose:
            x = np.arange(len(train_loss_list))
            plt.figure(figsize=(10, 6))
            plt.plot(x, np.array(train_loss_list), "b", label="Train Loss")
            plt.legend()  # 显示图例
            plt.xlabel("Epoch")  # 设置横坐标轴标签
            plt.ylabel("Loss")  # 设置纵坐标轴标签
            plt.savefig(os.path.join(path, f"{self.args.sub_name}_fold{fold_i + 1}_Loss_figure"))

        return self.model

    def test(self, setting, fold_i, train_val_set, test_set, result_path, test=0):
        # _, test_set = load_preprocessed_data(self.args)
        test_loader = dataloader(test_set[0], test_set[1], batch_size=self.args.batch_size, shuffle=False)
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints, setting,
                                        f'{self.args.sub_name}_fold{fold_i + 1}_checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(test_loader):
                batch_x = batch_x.to(self.device)
                label = label.to(self.device)

                outputs, _ = self.model(batch_x)
                outputs = F.softmax(outputs, dim=-1)

                preds.append(outputs)
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        # print('test shape:', preds.shape, trues.shape)

        predictions = torch.argmax(preds, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.cpu().numpy()
        auc = cal_auc(trues, preds[:, 1].cpu().numpy())
        ba = cal_ba(trues, predictions)
        f1_score = cal_F1_score(trues, predictions)
        conf_matrix = cal_confusion_matrix(trues, predictions)

        # result save
        # folder_path = os.path.join('./results/', setting)
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        print('{} AUC:{} BA:{} conf_matrix:{}'.format(self.args.sub_name, auc.round(4), ba.round(4), conf_matrix))
        # file_name = 'result_classification.txt'
        with open(result_path, 'a') as f:
            f.write(setting + "  \n")
            f.write('AUC:{}'.format(auc.round(4)))
            f.write('BA:{}'.format(ba.round(4)))
            f.write('confusion_matrix:{}'.format(conf_matrix))
            f.write('\n')
            f.write('\n')
        return auc.round(4), ba.round(4), f1_score.round(4), conf_matrix
