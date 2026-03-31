import torch
import torch.nn as nn
import numpy as np
import os
from Deep_learning.exp_basic import ExpBasic
from Data_Processing.make_dataset import load_preprocessed_data, dataloader
from Deep_learning.tools import EarlyStopping, cal_auc, cal_ba, cal_F1_score, cal_confusion_matrix, \
    plot_train_val_figure, EarlyStopping_with_C
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt


class RsvpClassification_SVDD_2stage(ExpBasic):
    def __init__(self, args):
        super(RsvpClassification_SVDD_2stage, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model
    # def _get_data(self):
    #     train_val_set, test_set = load_preprocessed_data(self.args)
    #     return train_val_set, test_set

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

    def _calculate_SVDD_loss(self, projection_dim, label, C):
        # 计算SVDD loss
        feature_distance = torch.norm(projection_dim - C, dim=1)
        targer_index = torch.where(label == 1)[0]
        nontarget_index = torch.where(label == 0)[0]
        supervised_distance = (feature_distance[targer_index] ** (-1)).sum() + \
                              (feature_distance[nontarget_index]).sum()
        return supervised_distance / len(label)

    def validate(self, vali_loader, criterion, C, stage=1):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(vali_loader):
                batch_x = batch_x.to(self.device)
                label = label.to(self.device)

                outputs, features = self.model(batch_x)
                # 取出90%的数据用于训练网络，剩下10%用于训练球心C

                pred = outputs.detach()
                ce_loss = criterion(pred, label.long())
                svdd_loss = self._calculate_SVDD_loss(features.detach(), label, C)
                loss = svdd_loss if stage == 1 else ce_loss
                total_loss.append(loss)

                preds.append(F.softmax(pred, dim=-1).detach())
                trues.append(label)

        total_loss = torch.tensor(total_loss).mean()

        if stage == 1:
            return total_loss, 0, 0
        elif stage == 2:
            preds = torch.cat(preds, 0)
            trues = torch.cat(trues, 0)
            predictions = torch.argmax(preds, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
            trues = trues.cpu().numpy()
            auc = cal_auc(trues, preds[:, 1].cpu().numpy())
            ba = cal_ba(trues, predictions)

            return total_loss, auc, ba

    def train(self, setting, fold_i, train_val_set, verbose=True, stage=1):
        # train_val_set, test_set = load_preprocessed_data(self.args)
        train_val_set_here = copy.deepcopy(train_val_set)
        val_x, val_y = train_val_set_here[0].pop(fold_i), train_val_set_here[1].pop(fold_i)
        train_x = np.concatenate(train_val_set_here[0], axis=0)
        train_y = np.concatenate(train_val_set_here[1], axis=0)
        train_loader = dataloader(train_x, train_y, batch_size=self.args.batch_size)
        val_loader = dataloader(val_x, val_y, batch_size=self.args.batch_size)
        class_weight = self._calculate_class_ratio(train_y)
        class_weight = torch.tensor(class_weight)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,
                                       path=os.path.join(path, f'{self.args.sub_name}_fold{fold_i+1}_checkpoint.pth'))

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(class_weight)
        criterion_C = nn.MSELoss()
        scheduler = ReduceLROnPlateau(model_optim, 'min', factor=0.5, patience=5, verbose=True)

        # SVDD参数球心C(dim=args.project_dim), 半径R
        # C = nn.Parameter(torch.randn(self.args.projection_dim, device=self.device).reshape(1, -1))
        self.model.eval()
        init_C = torch.empty(self.args.projection_dim, device = self.device).normal_(0, 0.1).reshape(1, -1)
        for i, (batch_x, label) in enumerate(train_loader):
            batch_x = batch_x.to(self.device)
            label = label.to(self.device)
            _, features = self.model(batch_x)
            nontarget_label = torch.where(label == 0)[0]
            center_C = features[nontarget_label]
            init_C = torch.cat((init_C, center_C), dim=0)
        C = nn.Parameter(init_C[1:].mean(dim=0).reshape(1, -1))
        # C_optimizer = torch.optim.Adagrad([C], lr=self.args.learning_rate, lr_decay=self.args.learning_rate // 10)

        train_loss_list, val_loss_list, val_auc_list, val_ba_list = [], [], [], []
        if stage == 1:
            # stage 1: representation learning
            for param in self.model.ClassifierBlock.parameters():
                param.requires_grad = False
        elif stage == 2:
            print(self.model.state_dict()['CNN_Block.1.weight'])
            # # stage 2: classifier learning
            for param in self.model.ClassifierBlock.parameters():
                param.requires_grad = True
            for name, param in self.model.named_parameters():
                if not name.startswith('ClassifierBlock'):
                    param.requires_grad = False
        else:
            raise ValueError("Invalid stage number!")
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
                outputs, features = self.model(batch_x)

                if stage == 1:
                    svdd_loss = self._calculate_SVDD_loss(features, label, C)
                    loss = svdd_loss
                    train_loss.append(loss.item())

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    model_optim.step()
                elif stage == 2:
                    loss = criterion(outputs, label.long())
                    train_loss.append(loss.item())

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    model_optim.step()

            train_loss = torch.tensor(train_loss).mean()
            val_loss, val_auc, val_ba = self.validate(val_loader, criterion, C, stage)
            # save training information
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            val_auc_list.append(val_auc)
            val_ba_list.append(val_ba)

            print(
                "{0} Fold{1}/{2} Epoch: {3}, learning rate: {4} | Train Loss: {5:.3f} Val Loss: {6:.3f} "
                "Val AUC: {7:.3f} Val BA: {8:.3f}"
                .format(self.args.sub_name, fold_i+1, self.args.n_fold, epoch + 1, current_lr, train_loss, val_loss,
                        val_auc, val_ba))
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 调整学习率
            scheduler.step(val_loss)

        # # save best model
        # best_model_path = os.path.join(path, f'{self.args.sub_name}_fold{fold_i+1}_checkpoint.pth')
        # self.model.load_state_dict(torch.load(best_model_path))


        # plot figure and save
        if verbose:
            plot_train_val_figure(
                train_loss_list, val_loss_list, val_auc_list, val_ba_list, path, fold_i+1, self.args.sub_name)

    def test(self, setting, fold_i, train_val_set, test_set, result_path, test=0):
        # _, test_set = load_preprocessed_data(self.args)
        test_loader = dataloader(test_set[0], test_set[1], batch_size=self.args.batch_size)
        print('loading model')
        self.model.load_state_dict(
            torch.load(os.path.join(self.args.checkpoints, setting, f'{self.args.sub_name}_fold{fold_i+1}_checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(test_loader):
                batch_x = batch_x.to(self.device)
                label = label.to(self.device)

                outputs, _ = self.model(batch_x)
                outputs = F.softmax(outputs, dim=-1)

                preds.append(outputs.detach())
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
