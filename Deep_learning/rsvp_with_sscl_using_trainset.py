import torch
import torch.nn as nn
import numpy as np
import os
from Deep_learning.exp_basic import ExpBasic
from Data_Processing.make_dataset import load_preprocessed_data, dataloader
from Deep_learning.tools import EarlyStopping, cal_auc, cal_ba, cal_F1_score, cal_confusion_matrix, \
    plot_train_val_figure, Gamma_reduce
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt


class RsvpClassification_using_trainset(ExpBasic):
    def __init__(self, args):
        super(RsvpClassification_using_trainset, self).__init__(args)

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
        # target_features_queue = Queue(max_size=20)
        # awl = AutomaticWeightedLoss(3)
        Sup_loss = SupConLoss()
        augmentor = DataAugmentation()
        train_val_set_here = copy.deepcopy(train_val_set)
        train_x = np.concatenate(train_val_set_here[0], axis=0)
        train_y = np.concatenate(train_val_set_here[1], axis=0)
        train_loader = dataloader(train_x, train_y, batch_size=self.args.batch_size)
        class_weight = torch.tensor(self._calculate_class_ratio(train_y))

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,
                                       path=os.path.join(path, f'{self.args.sub_name}_fold{fold_i+1}_checkpoint.pth'))
        gamma_regulation = Gamma_reduce(patience=10)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(class_weight)
        scheduler = ReduceLROnPlateau(model_optim, 'min', factor=0.5, patience=5, verbose=True)

        train_loss_list = []
        train_ce_loss_list, train_sc_loss_list = [], []
        train_p_loss_list = []
        gamma = 0.1
        for epoch in range(self.args.train_epochs):
            current_lr = model_optim.param_groups[0]["lr"]
            train_loss = []
            train_ce_loss, train_sc_loss = [], []
            train_p_loss = []
            target_list = []
            # target_features_queue.clear()

            self.model.train()
            # if epoch >= 50:
            #     gamma = 0.01

            for i, (batch_x, label) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.to(self.device)
                label = label.to(self.device)
                target_index = torch.where(label == 1)[0]
                nontarget_index = torch.where(label == 0)[0]
                if len(target_index) == 0:
                    continue

                # 数据增强
                batch_x_aug = augmentor.augment(batch_x)   # list of augmented data [list1, list2]
                _, features_aug = self.model(torch.cat(batch_x_aug, dim=0))
                f1, f2 = torch.split(features_aug, batch_x.shape[0], dim=0)
                features_aug = torch.stack((f1, f2), dim=1)

                outputs, features = self.model(batch_x)
                # for index in target_index:
                #     target_features_queue.push(features[index].unsqueeze(0).detach())
                # target_prototype = torch.cat(target_features_queue.to_list(), dim=0).mean(dim=0, keepdim=True)
                ce_loss = criterion(outputs, label.long())
                sup_loss = Sup_loss(features_aug)
                # p_loss = torch.exp((features @ target_prototype.T).mean())
                loss = ce_loss + sup_loss * gamma
                # loss = awl(sup_loss, p_loss, ce_loss)
                train_loss.append(loss.item())
                train_ce_loss.append(ce_loss.item())
                train_sc_loss.append(sup_loss.item())
                # train_p_loss.append(p_loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            train_loss = torch.tensor(train_loss).mean()
            train_ce_loss = torch.tensor(train_ce_loss).mean()
            train_sc_loss = torch.tensor(train_sc_loss).mean()
            train_p_loss = torch.tensor(train_p_loss).mean()

            # save training information
            train_loss_list.append(train_loss)
            train_ce_loss_list.append(train_ce_loss)
            train_sc_loss_list.append(train_sc_loss)
            train_p_loss_list.append(train_p_loss)

            print(
                "{0} Fold{1}/{2} Epoch: {3}, learning rate: {4} | Train Loss: {5:.3f}"
                .format(self.args.sub_name, fold_i+1, self.args.n_fold, epoch + 1, current_lr, train_loss))
            early_stopping(train_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # if not gamma_regulation.gamma_reduce:
            #     gamma_regulation(val_sc_loss)
            # else:
            #     gamma = 0.01

            # 调整学习率
            scheduler.step(train_loss)

        # save best model
        best_model_path = os.path.join(path, f'{self.args.sub_name}_fold{fold_i+1}_checkpoint.pth')
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
            plt.clf()
            plt.plot(x, np.array(train_ce_loss_list), "b", label="Train CE Loss")
            plt.legend()  # 显示图例
            plt.xlabel("Epoch")  # 设置横坐标轴标签
            plt.ylabel("Loss")  # 设置纵坐标轴标签
            plt.savefig(os.path.join(path, f"{self.args.sub_name}_fold{fold_i+1}_CE_Loss_figure"))
            plt.clf()
            plt.plot(x, np.array(train_sc_loss_list), "b", label="Train SC Loss")
            plt.legend()  # 显示图例
            plt.xlabel("Epoch")  # 设置横坐标轴标签
            plt.ylabel("Loss")  # 设置纵坐标轴标签
            plt.savefig(os.path.join(path, f"{self.args.sub_name}_fold{fold_i + 1}_SC_Loss_figure"))
            plt.close()

        return self.model

    def test(self, setting, fold_i, train_val_set, test_set, result_path, test=0):
        # _, test_set = load_preprocessed_data(self.args)
        test_loader = dataloader(test_set[0], test_set[1], batch_size=self.args.batch_size, shuffle=False)
        if test:
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


# class SupConLoss(nn.Module):
#     """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
#     It also supports the unsupervised contrastive loss in SimCLR"""
#     def __init__(self, temperature=0.07, base_temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.base_temperature = base_temperature
#
#     def forward(self, features, labels=None, mask=None):
#         """Compute loss for model. If both `labels` and `mask` are None,
#         it degenerates to SimCLR unsupervised loss:
#         https://arxiv.org/pdf/2002.05709.pdf
#
#         Args:
#             features: hidden vector of shape [bsz, n_features].
#             labels: ground truth of shape [bsz].
#             mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
#                 has the same class as sample i. Can be asymmetric.
#         Returns:
#             A loss scalar.
#         """
#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))
#
#         target_indices = torch.where(labels == 1)[0]
#         if len(target_indices) == 1:
#             features = torch.cat([features[0:target_indices], features[target_indices+1:]], dim=0)
#             labels = torch.cat([labels[0:target_indices], labels[target_indices+1:]], dim=0)
#         batch_size = features.shape[0]
#         labels = labels.reshape(-1, 1)
#         mask = torch.eq(labels, labels.T).float().to(device)  # (bsz, bsz)跟每行代表的样本同标签的位置为1，包括自身
#
#         # compute logits
#         anchor_dot_contrast = torch.div(
#             torch.matmul(features, features.T),
#             self.temperature)
#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()
#
#         # mask-out self-contrast cases
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(batch_size).reshape(-1, 1).to(device),
#             0
#         )
#         mask = mask * logits_mask
#
#         # compute log_prob
#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # 这里的logits=log(exp(logits))，然后-相当于/
#
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
#
#         # loss
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
#         loss = loss.reshape(1, batch_size).mean()
#
#         return loss
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class DataAugmentation:
    def __init__(self):
        pass

    def signflip(self, data):
        return data * -1

    def add_noise(self, data):
        batch_size, _, num_channels, num_time_points = data.shape
        noise = torch.randn(batch_size, 1, num_channels, num_time_points // 5, device=data.device)
        data[:, :, :, -num_time_points // 5:-1] += noise
        return data

    def add_noise_P3(self, data):
        p3_region = [38, 64]
        noise = torch.randn(data.shape[0], 1, data.shape[2], p3_region[1] - p3_region[0], device=data.device)
        data[:, :, :, p3_region[0]:p3_region[1]] += noise
        return data

    def replace_with_noise(self, data):
        noise = torch.randn(data.shape[0], 1, data.shape[2], data.shape[3] // 4, device=data.device)
        data[:, :, :, -data.shape[3] // 4:] = noise
        return data

    def shuffle_channels(self, data):
        for i in range(data.shape[0]):
            data[i, 0] = data[i, 0][torch.randperm(data.shape[2])]
        return data

    def augment(self, data):
        return [
            self.signflip(data.clone()),
            self.add_noise(data.clone())
        ]

# class DataAugmentation:
#     def __init__(self, probability=1.0, std=0.1, random_state=None):
#         self.probability = probability
#         self.std = std
#         self.random_state = random_state
#         self.rng = np.random.default_rng(random_state)
#
#     def signflip(self, data):
#         """
#         Perform sign flip on the time domain of the data.
#         """
#         batch_size = data.shape[0]
#         augmented_data = data.clone()
#         num_augmented = int(batch_size * self.probability)
#         indices = self.rng.choice(batch_size, num_augmented, replace=False)
#         augmented_data[indices] *= -1
#         return augmented_data
#
#     def add_noise(self, data):
#         """
#         Add Gaussian noise to the last 1/4 of the time domain.
#         """
#         batch_size, _, num_channels, num_time_points = data.shape
#         augmented_data = data.clone()
#         num_augmented = int(batch_size * self.probability)
#         indices = self.rng.choice(batch_size, num_augmented, replace=False)
#         noise = torch.randn(len(indices), 1, num_channels, num_time_points // 5, device=data.device) * self.std
#         augmented_data[indices, :, :, -num_time_points // 5:-1] += noise
#         return augmented_data
#
#     def add_noise_P3(self, data):
#         """
#         Add Gaussian noise to the last 1/5 of the time domain.
#         """
#         p3_region = [38, 64]
#         batch_size, _, num_channels, num_time_points = data.shape
#         augmented_data = data.clone()
#         num_augmented = int(batch_size * self.probability)
#         indices = self.rng.choice(batch_size, num_augmented, replace=False)
#         noise = torch.randn(len(indices), 1, num_channels, 64 - 38, device=data.device) * self.std
#         augmented_data[indices, :, :, 38:64] += noise
#         return augmented_data
#
#     def replace_with_noise(self, data):
#         """
#         Replace the last 1/4 of the time domain with Gaussian noise.
#         """
#         batch_size, _, num_channels, num_time_points = data.shape
#         augmented_data = data.clone()
#         num_augmented = int(batch_size * self.probability)
#         indices = self.rng.choice(batch_size, num_augmented, replace=False)
#         noise = torch.randn(len(indices), 1, num_channels, num_time_points // 4, device=data.device) * self.std
#         augmented_data[indices, :, :, -num_time_points // 4:] = noise
#         return augmented_data
#
#     def shuffle_channels(self, data):
#         """
#         Randomly shuffle the channels of the data.
#         """
#         batch_size, _, num_channels, num_time_points = data.shape
#         augmented_data = data.clone()
#         num_augmented = int(batch_size * self.probability)
#         indices = self.rng.choice(batch_size, num_augmented, replace=False)
#         for i in indices:
#             augmented_data[i, 0] = augmented_data[i, 0][torch.randperm(num_channels)]
#         return augmented_data
#
#     def augment(self, data):
#         """
#         Apply the augmentations and return a list of augmented data.
#         """
#         signflipped_data = self.signflip(data)
#         noise_added_data = self.add_noise(data)
#         noise_replaced_data = self.replace_with_noise(data)
#         shuffled_data = self.shuffle_channels(data)
#         noise_added_p3 = self.add_noise_P3(data)
#
#         return [signflipped_data, noise_added_data]


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


class Queue(object):
    def __init__(self, max_size=18):
        self.max_size = max_size
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def push(self, item):
        if len(self.data) < self.max_size:
            self.data.append(item)
        else:
            self.data.pop(0)
            self.data.append(item)

    def clear(self):
        self.data = []

    def to_list(self):
        return self.data
