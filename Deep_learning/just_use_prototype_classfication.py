import torch
import torch.nn as nn
import numpy as np
import os
from Deep_learning.exp_basic import ExpBasic
from Data_Processing.make_dataset import SupportQuerySplit, dataloader, SupportQuerySplitSeparately
from Deep_learning.tools import EarlyStopping, cal_auc, cal_ba, cal_F1_score, cal_confusion_matrix, \
    plot_train_val_figure
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import torch.nn.functional as F


class Just_PrototypeClassification(ExpBasic):
    def __init__(self, args):
        super(Just_PrototypeClassification, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _select_optimizer(self):
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                       betas=(self.args.beta1, self.args.beta2), weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def compute_prototypes(self, support_features, Ns0) -> torch.Tensor:
        """Compute class prototypes from support samples."""
        class_prototype0 = support_features[:Ns0].mean(dim=0, keepdim=True)
        class_prototype1 = support_features[Ns0:].mean(dim=0, keepdim=True)
        class_prototypes = torch.cat((class_prototype0, class_prototype1), dim=0)
        return class_prototypes

    def cal_distances(self, query_vectors, prototype_vectors, mode="l2") -> torch.Tensor:
        """
        calculate distances between query samples and prototypes
        :param query_vectors: (Nq*2, features)
        :param prototype_vectors: (2, features)
        :param mode: l2 or cosine or inner_product
        :return: distances (Nq, 2)
        """
        Nq = query_vectors.shape[0]
        n_class = prototype_vectors.shape[0]

        if mode == "inner_product":
            distance = torch.matmul(query_vectors, prototype_vectors.T)
        elif mode == "l2":
            distance = (query_vectors.unsqueeze(1).expand(Nq, n_class, -1) -
                        prototype_vectors.unsqueeze(0).expand(Nq, n_class, -1)
                        ).pow(2).sum(dim=-1)
        elif mode == "cosine_similarity":
            distance = torch.cosine_similarity(
                query_vectors.unsqueeze(1).expand(Nq, n_class, -1),
                prototype_vectors.unsqueeze(0).expand(Nq, n_class, -1),
                dim=-1)
        else:
            raise ValueError("Invalid mode!")
        return distance

    def validate(self, train_set, val_set, criterion):
        val_Ns0 = self.args.val_Ns * 2
        val_Ns1 = self.args.val_Ns
        support_model = SupportQuerySplitSeparately(
            train_set[0],
            train_set[1],
            1, val_Ns0, val_Ns1, 1)
        support_set_x, support_set_y, _ = support_model.support_query_set_split(mode="validate")
        support_samples = support_set_x[:(val_Ns0 + val_Ns1)]
        support_samples = torch.from_numpy(support_samples).unsqueeze(dim=1)
        val_loader = dataloader(val_set[0], val_set[1], batch_size=val_set[0].shape[0])
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(val_loader):
                # 处理每个批量数据
                batch_x = batch_x.to(self.device)
                label = label.to(self.device)

                # Embed all samples
                _, support_embeddings = self.model(support_samples.to(self.device))
                prototypes = self.compute_prototypes(support_embeddings, val_Ns0)
                _, query_embeddings = self.model(batch_x)
                distances = self.cal_distances(query_embeddings, prototypes, mode="l2")  # (Nq, 2)

                ce_loss = criterion(-distances, label.long())
                loss = ce_loss

                preds = F.softmax(-distances, dim=-1).detach()

        predictions = torch.argmax(preds, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = label.cpu().numpy()
        auc = cal_auc(trues, preds[:, 1].cpu().numpy())
        ba = cal_ba(trues, predictions)

        return loss.detach().cpu().numpy(), auc, ba

    def train(self, setting, fold_i, train_val_set, verbose=True):
        # 输入train_val_set，划分训练集和验证集之后，还要分别划分支持集和查询集
        train_val_set_here = copy.deepcopy(train_val_set)
        val_x, val_y = train_val_set_here[0].pop(fold_i), train_val_set_here[1].pop(fold_i)
        train_x = np.concatenate(train_val_set_here[0], axis=0)
        train_y = np.concatenate(train_val_set_here[1], axis=0)
        # 对训练集划分支持集和查询集
        support_query_model = SupportQuerySplitSeparately(train_x, train_y, self.args.episodes, self.args.train_Ns * 2,
                                                          self.args.train_Ns, self.args.train_Nq)
        train_set_x, train_set_y, train_batch_size = support_query_model.support_query_set_split(mode="train")
        train_loader = dataloader(train_set_x, train_set_y, batch_size=train_batch_size, shuffle=False)

        # 划分验证集的支持集和查询集，以免每个episode变化
        # test_support_query_model = SupportQuerySplit(val_x, val_y, 1, self.args.val_Ns, self.args.val_Nq)
        # val_set_x, val_set_y, val_batch_size = test_support_query_model.support_query_set_split(mode="validate")
        # val_loader = dataloader(val_set_x, val_set_y, batch_size=val_batch_size, shuffle=False)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,
                                       path=os.path.join(path, f'{self.args.sub_name}_fold{fold_i+1}_checkpoint.pth'))

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = ReduceLROnPlateau(model_optim, 'min', factor=0.5, patience=10, verbose=True)

        train_loss_list, val_loss_list, val_auc_list, val_ba_list = [], [], [], []
        for epoch in range(self.args.train_epochs):
            current_lr = model_optim.param_groups[0]["lr"]
            train_loss = []

            self.model.train()

            for i, (batch_x, label) in enumerate(train_loader):
                model_optim.zero_grad()
                # 处理每个批量数据
                batch_x = batch_x.to(self.device)
                label = label.to(self.device)

                # Embed all samples
                _, embeddings = self.model(batch_x)
                support_embeddings = embeddings[:3*self.args.train_Ns]
                support_label = label[:3*self.args.train_Ns]
                query_embeddings = embeddings[3*self.args.train_Ns:]
                query_label = label[3*self.args.train_Ns:]

                prototypes = self.compute_prototypes(support_embeddings, self.args.train_Ns*2)

                # 计算查询集和原型向量直接的距离，0是非目标，1是目标
                distances = self.cal_distances(query_embeddings, prototypes, mode="l2")  # (Nq*2, 2)

                ce_loss = criterion(-distances, query_label.long())
                # 现在增加一个metric loss 就是用相同类别的距离-不同类别的距离
                # 首先在query_prototype_scores的基础上用非目标-目标
                query_prototype_difference = distances[:, 0] - distances[:, 1]  # (Nq, )
                # 把label=1的不变，label=0的取负
                correct_label = query_label * (-2) + 1  # (Nq, )
                metric_loss = query_prototype_difference * correct_label
                loss = ce_loss
                train_loss.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            train_loss = torch.tensor(train_loss).mean()
            val_loss, val_auc, val_ba = self.validate([train_x, train_y], [val_x, val_y], criterion)
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

        # save best model
        best_model_path = os.path.join(path, f'{self.args.sub_name}_fold{fold_i+1}_checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        # plot figure and save
        if verbose:
            plot_train_val_figure(
                train_loss_list, val_loss_list, val_auc_list, val_ba_list, path, fold_i+1, self.args.sub_name)

        return self.model

    def test(self, setting, fold_i, train_val_set, test_set, result_path, test=0):
        """
        如果 n_shot==0 测试集直接用训练集抽取每类 train_Ns 个样本做支持集
        如果 n_shot>0 测试集用测试集抽取每类 n_shot 个样本做支持集
        """
        test_x, test_y = test_set[0], test_set[1]
        if self.args.n_shot == 0:
            test_Ns = self.args.train_Ns*1
            support_model = SupportQuerySplitSeparately(
                np.concatenate(train_val_set[0]),
                np.concatenate(train_val_set[1]),
                1, test_Ns*2, test_Ns, 1)
            support_set_x, support_set_y, _ = support_model.support_query_set_split(mode="validate")
            support_samples = support_set_x[:3*test_Ns]
        else:
            test_Ns = self.args.n_shot
            support_model = SupportQuerySplitSeparately(test_x, test_y, 1, test_Ns*2, test_Ns, None)
            support_set_x, support_set_y, _ = support_model.support_query_set_split(mode="validate")
            support_samples = support_set_x[:2 * self.args.n_shot]
            test_x = support_set_x[2 * self.args.n_shot:]
            test_y = support_set_y[2 * self.args.n_shot:]
        support_samples = torch.from_numpy(support_samples).unsqueeze(dim=1)
        test_loader = dataloader(test_x, test_y, batch_size=self.args.batch_size)
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(
                    self.args.checkpoints, setting, f'{self.args.sub_name}_fold{fold_i+1}_checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(test_loader):
                batch_x = batch_x.to(self.device)
                label = label.to(self.device)

                _, support_embeddings = self.model(support_samples.to(self.device))
                prototypes = self.compute_prototypes(support_embeddings, test_Ns*2)
                _, query_embeddings = self.model(batch_x)
                distances = self.cal_distances(query_embeddings, prototypes, mode="l2")  # (Nq, 2)
                probs = F.softmax(-distances, dim=-1)

                preds.append(probs.detach())
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
