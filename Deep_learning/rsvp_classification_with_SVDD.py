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


class RsvpClassification_with_SVDD(ExpBasic):
    def __init__(self, args):
        super(RsvpClassification_with_SVDD, self).__init__(args)

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

    def _calculate_gamma(self, data_loader, criterion, C):
        # 计算CE loss和SVDD loss的比例
        init_CE_loss, init_SVDD_loss = 0, 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(data_loader):
                batch_x = batch_x.to(self.device)
                label = label.to(self.device)

                outputs, features = self.model(batch_x)
                outputs = outputs.detach()
                features = features.detach()
                CE_loss = criterion(outputs, label.long())
                SVDD_loss = self._calculate_SVDD_loss(features, label, 0)
                init_CE_loss += CE_loss
                init_SVDD_loss += SVDD_loss
        return init_CE_loss / init_SVDD_loss

    def validate(self, vali_loader, criterion, C, gamma):
        total_loss = []
        total_ce_loss, total_svdd_loss = [], []
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
                loss = ce_loss + gamma * svdd_loss
                total_loss.append(loss)
                total_ce_loss.append(ce_loss)
                total_svdd_loss.append(svdd_loss)

                preds.append(F.softmax(pred, dim=-1).detach())
                trues.append(label)

        total_loss = torch.tensor(total_loss).mean()
        total_ce_loss = torch.tensor(total_ce_loss).mean()
        total_svdd_loss = torch.tensor(total_svdd_loss).mean()

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        predictions = torch.argmax(preds, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.cpu().numpy()
        auc = cal_auc(trues, preds[:, 1].cpu().numpy())
        ba = cal_ba(trues, predictions)

        return total_loss, total_ce_loss, total_svdd_loss, auc, ba

    def train(self, setting, fold_i, train_val_set, verbose=True):
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
        early_stopping = EarlyStopping_with_C(patience=self.args.patience, verbose=True,
                                       path=os.path.join(path, f'{self.args.sub_name}_fold{fold_i+1}_checkpoint.pth'))

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(class_weight)
        criterion_C = nn.MSELoss()
        scheduler = ReduceLROnPlateau(model_optim, 'min', factor=0.5, patience=5, verbose=True)

        # SVDD参数球心C(dim=args.project_dim), 半径R
        C = nn.Parameter(torch.randn(self.args.projection_dim, device=self.device).reshape(1, -1))
        # self.model.eval()
        # init_C = torch.empty(self.args.projection_dim, device = self.device).normal_(0, 0.1).reshape(1, -1)
        # for i, (batch_x, label) in enumerate(train_loader):
        #     batch_x = batch_x.to(self.device)
        #     label = label.to(self.device)
        #     _, features = self.model(batch_x)
        #     nontarget_label = torch.where(label == 0)[0]
        #     center_C = features[nontarget_label]
        #     init_C = torch.cat((init_C, center_C), dim=0)
        # C = nn.Parameter(init_C[1:].mean(dim=0).reshape(1, -1))
        C_optimizer = torch.optim.Adagrad([C], lr=self.args.learning_rate, lr_decay=self.args.learning_rate // 10)
        gamma = self._calculate_gamma(train_loader, criterion, C)

        train_loss_list, val_loss_list, val_auc_list, val_ba_list = [], [], [], []
        train_ce_loss_list, train_svdd_loss_list = [], []
        val_ce_loss_list, val_svdd_loss_list = [], []
        for epoch in range(self.args.train_epochs):
            current_lr = model_optim.param_groups[0]["lr"]
            train_loss = []
            train_ce_loss, train_svdd_loss = [], []

            self.model.train()

            for i, (batch_x, label) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.to(self.device)
                label = label.to(self.device)
                target_label = torch.where(label == 1)[0]
                if len(target_label) == 0:
                    continue

                # 取出90%的数据用于训练网络，剩下10%用于训练球心C
                num_for_net = int(len(batch_x) * 0.9)
                batch_x_for_net, batch_x_for_C = batch_x[:num_for_net], batch_x[num_for_net:]
                outputs_for_net, features_for_net = self.model(batch_x_for_net)
                label_for_net, label_for_C = label[:num_for_net], label[num_for_net:]
                ce_loss = criterion(outputs_for_net, label_for_net.long())
                svdd_loss = self._calculate_SVDD_loss(features_for_net, label_for_net, C)
                loss = ce_loss + gamma * svdd_loss
                train_loss.append(loss.item())
                train_ce_loss.append(ce_loss.item())
                train_svdd_loss.append(svdd_loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

                # 更新球心C
                C_optimizer.zero_grad()
                nontarget_label_for_C = torch.where(label_for_C == 0)[0]
                _, features_for_C = self.model(batch_x_for_C)
                center = features_for_C[nontarget_label_for_C].mean(dim=0).reshape(1, -1)
                loss_C = criterion_C(C, center)
                loss_C.backward()
                C_optimizer.step()

            train_loss = torch.tensor(train_loss).mean()
            train_ce_loss = torch.tensor(train_ce_loss).mean()
            train_svdd_loss = torch.tensor(train_svdd_loss).mean()
            val_loss, val_ce_loss, val_svdd_loss, val_auc, val_ba = self.validate(val_loader, criterion, C, gamma)
            # save training information
            train_loss_list.append(train_loss)
            train_ce_loss_list.append(train_ce_loss)
            train_svdd_loss_list.append(train_svdd_loss)
            val_loss_list.append(val_loss)
            val_ce_loss_list.append(val_ce_loss)
            val_svdd_loss_list.append(val_svdd_loss)
            val_auc_list.append(val_auc)
            val_ba_list.append(val_ba)

            print(
                "{0} Fold{1}/{2} Epoch: {3}, learning rate: {4} | Train Loss: {5:.3f} Val Loss: {6:.3f} "
                "Val AUC: {7:.3f} Val BA: {8:.3f}"
                .format(self.args.sub_name, fold_i+1, self.args.n_fold, epoch + 1, current_lr, train_loss, val_loss,
                        val_auc, val_ba))
            early_stopping(val_loss, self.model, C)
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
            x = np.arange(len(train_ce_loss_list))
            plt.figure(figsize=(10, 6))
            plt.plot(x, np.array(train_ce_loss_list), "b", label="Train CE Loss")
            plt.plot(x, np.array(val_ce_loss_list), "g", label="Val CE Loss")
            plt.legend()  # 显示图例
            plt.xlabel("Epoch")  # 设置横坐标轴标签
            plt.ylabel("Loss")  # 设置纵坐标轴标签
            plt.savefig(os.path.join(path, f"{self.args.sub_name}_fold{fold_i + 1}_CE_Loss_figure"))
            plt.clf()
            plt.plot(x, np.array(train_svdd_loss_list), "b", label="Train SVDD Loss")
            plt.plot(x, np.array(val_svdd_loss_list), "g", label="Val SVDD Loss")
            plt.legend()  # 显示图例
            plt.xlabel("Epoch")  # 设置横坐标轴标签
            plt.ylabel("Loss")  # 设置纵坐标轴标签
            plt.savefig(os.path.join(path, f"{self.args.sub_name}_fold{fold_i + 1}_SVDD_Loss_figure"))
            plt.close()

    def test(self, setting, fold_i, train_val_set, test_set, result_path, test=0):
        # _, test_set = load_preprocessed_data(self.args)
        test_loader = dataloader(test_set[0], test_set[1], batch_size=self.args.batch_size)
        print('loading model')
        self.model.load_state_dict(
            torch.load(os.path.join(self.args.checkpoints, setting, f'{self.args.sub_name}_fold{fold_i+1}_checkpoint.pth')))
        C = torch.load(os.path.join(self.args.checkpoints, setting, f'{self.args.sub_name}_fold{fold_i+1}_checkpoint_C.pth'))

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
