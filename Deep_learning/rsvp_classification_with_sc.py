import torch
import torch.nn as nn
import numpy as np
import os
from Deep_learning.exp_basic import ExpBasic
from Data_Processing.make_dataset import dataloader
from Deep_learning.tools import EarlyStopping, cal_auc, cal_ba, cal_F1_score, cal_confusion_matrix, \
    plot_train_val_figure, calculate_tpr_tnr
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LambdaLR
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import math


class RsvpClassification_with_sc(ExpBasic):
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _select_optimizer(self):
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                       weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self, y_data):
        target_num = len(np.where(y_data == 1)[0])
        nontarget_num = len(np.where(y_data == 0)[0])
        class_weight = torch.tensor([1, (nontarget_num / target_num)]).to(self.device)
        return nn.CrossEntropyLoss(weight=class_weight)

    def _augment_data(self, batch_x, augmentor):
        batch_x_aug = augmentor.augment(batch_x)  # list of augmented data [list1, list2]
        _, features_aug = self.model(torch.cat(batch_x_aug, dim=0))
        f1, f2 = torch.split(features_aug, batch_x.shape[0], dim=0)
        return torch.stack((f1, f2), dim=1)

    def plot_and_save(self, x, y_list, labels, xlabel, ylabel, filename):
        plt.figure(figsize=(10, 6))
        for y, label in zip(y_list, labels):
            plt.plot(x, y, label=label)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(filename)
        plt.clf()

    def validate(self, vali_loader, criterion, criterion_sc, augmentor, gamma=0.5):
        preds = []
        total_outputs, total_label, total_feature_aug = [], [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(vali_loader):
                batch_x = batch_x.to(self.device)
                label = label.to(self.device)
                features_aug = self._augment_data(batch_x, augmentor)

                outputs, _ = self.model(batch_x)
                preds.append(F.softmax(outputs, dim=-1))
                total_outputs.append(outputs)
                total_label.append(label)
                total_feature_aug.append(features_aug)

            total_outputs = torch.cat(total_outputs, 0)
            total_label = torch.cat(total_label, 0)
            total_feature_aug = torch.cat(total_feature_aug, 0)

            ce_loss = criterion(total_outputs, total_label.long())
            sc_loss = criterion_sc(total_feature_aug)
            loss = ce_loss + sc_loss * gamma

        preds = torch.cat(preds, 0)
        predictions = torch.argmax(preds, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = total_label.cpu().numpy()
        auc = cal_auc(trues, preds[:, 1].cpu().numpy())
        ba = cal_ba(trues, predictions)
        tpr, tnr = calculate_tpr_tnr(trues, predictions)

        return loss.cpu().numpy(), ce_loss.cpu().numpy(), sc_loss.cpu().numpy(), auc, ba, tpr, tnr

    def train(self, setting, fold_i, train_val_set, verbose=True):
        # conv1_weights = self.model.CNN_Block[1].weight.data
        # # 查看权重
        # print(conv1_weights)
        Sup_loss = SupConLoss(temperature=0.05, base_temperature=0.05)
        augmentor = DataAugmentation()
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
        # scheduler = CosineAnnealingLR(model_optim, T_max=10, eta_min=0)
        # warm_up_iter = 20
        # lr_max = 0.0001
        # lr_min = 1e-8
        # T_max = 35
        # lambda0 = lambda cur_iter: 0.1 if cur_iter < warm_up_iter else \
        #     (lr_min + 0.5 * (lr_max - lr_min) * (
        #                 1.0 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi))) / 0.0001
        # scheduler = LambdaLR(model_optim, lr_lambda=lambda0)

        train_loss_list, val_loss_list, val_auc_list, val_ba_list = [], [], [], []
        val_tpr_list, val_tnr_list = [], []
        train_ce_loss_list, train_sc_loss_list = [], []
        val_ce_loss_list, val_sc_loss_list = [], []
        gamma = 0.1
        lr_list = []
        for epoch in range(self.args.train_epochs):
            # if epoch <10:
            #     model_optim.param_groups[0]["lr"] = 1e-5
            # if epoch == 10:
            #     model_optim.param_groups[0]["lr"] = 1e-4
            current_lr = model_optim.param_groups[0]["lr"]
            lr_list.append(current_lr)
            train_loss = []
            train_ce_loss, train_sc_loss = [], []

            self.model.train()

            for i, (batch_x, label) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.to(self.device)
                label = label.to(self.device)
                target_index = torch.where(label == 1)[0]
                nontarget_index = torch.where(label == 0)[0]
                if len(target_index) == 0:
                    continue

                outputs, _ = self.model(batch_x)
                # 数据增强投影
                features_aug = self._augment_data(batch_x, augmentor)

                ce_loss = criterion(outputs, label.long())
                sup_loss = Sup_loss(features_aug)
                loss = ce_loss + sup_loss * gamma
                train_loss.append(loss.item())
                train_ce_loss.append(ce_loss.item())
                train_sc_loss.append(sup_loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            train_loss = sum(train_loss) / len(train_loss)
            train_ce_loss = sum(train_ce_loss) / len(train_ce_loss)
            train_sc_loss = sum(train_sc_loss) / len(train_sc_loss)

            # validation
            val_loss, val_ce_loss, val_sc_loss, val_auc, val_ba, val_tpr, val_tnr = self.validate(val_loader, criterion, Sup_loss, augmentor, gamma)

            # save training information
            train_loss_list.append(train_loss)
            train_ce_loss_list.append(train_ce_loss)
            train_sc_loss_list.append(train_sc_loss)
            val_loss_list.append(val_loss)
            val_ce_loss_list.append(val_ce_loss)
            val_sc_loss_list.append(val_sc_loss)
            val_auc_list.append(val_auc)
            val_ba_list.append(val_ba)
            val_tpr_list.append(val_tpr)
            val_tnr_list.append(val_tnr)

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
            # if epoch > 10:
            #     scheduler.step(val_loss)
            scheduler.step(val_loss)

        # save best model
        best_model_path = os.path.join(path, f'{self.args.sub_name}_fold{fold_i+1}_checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        # plot figure and save
        if verbose:
            plot_train_val_figure(
                train_loss_list, val_loss_list, val_auc_list, val_ba_list, path, fold_i+1, self.args.sub_name)
            x = np.arange(len(train_ce_loss_list))
            self.plot_and_save(x, [train_ce_loss_list, val_ce_loss_list], ['Train CE Loss', 'Val CE Loss'],
                               'Epoch', 'Loss',
                               os.path.join(path, f"{self.args.sub_name}_fold{fold_i+1}_CE_Loss_figure"))
            self.plot_and_save(x, [train_sc_loss_list, val_sc_loss_list], ['Train SC Loss', 'Val SC Loss'],
                               'Epoch', 'Loss',
                               os.path.join(path, f"{self.args.sub_name}_fold{fold_i+1}_SC_Loss_figure"))
            self.plot_and_save(x, [val_tpr_list, val_tnr_list], ['Val TPR', 'Val TNR'],
                               'Epoch', 'TPR/TNR',
                               os.path.join(path, f"{self.args.sub_name}_fold{fold_i+1}_TPR_TNR_figure"))

        return self.model

    def test(self, setting, fold_i, test_set, result_path):
        test_loader = dataloader(test_set[0], test_set[1], batch_size=self.args.batch_size, shuffle=False)
        print('loading model')
        self.model.load_state_dict(
            torch.load(os.path.join('.\\checkpoints', setting, f'{self.args.sub_name}_fold{fold_i+1}_checkpoint.pth')))

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
        tpr, tnr = calculate_tpr_tnr(trues, predictions)

        print('{} AUC:{} BA:{} conf_matrix:{}'.format(self.args.sub_name, auc.round(4), ba.round(4), conf_matrix))
        # file_name = 'result_classification.txt'
        with open(result_path, 'a') as f:
            f.write(setting + "  \n")
            f.write('AUC:{}'.format(auc.round(4)))
            f.write('BA:{}'.format(ba.round(4)))
            f.write('confusion_matrix:{}'.format(conf_matrix))
            f.write('TPR:{}'.format(tpr))
            f.write('TNR:{}'.format(tnr))
            f.write('\n')
            f.write('\n')
        return auc.round(4), ba.round(4), f1_score.round(4), conf_matrix, tpr.round(4), tnr.round(4)


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

    def bandpass_filter(self, data, lowcut=0.5, highcut=7.0, fs=128, order=5):
        def butter_bandpass(lowcut, highcut, fs, order=4):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            [b, a] = butter(order, [low, high], btype='bandpass')
            return b, a

        b, a = butter_bandpass(lowcut, highcut, fs, order)
        filtered_data = [filtfilt(b, a, trial, axis=-1) for trial in data[:, 0].cpu().numpy()]
        return torch.tensor(filtered_data, dtype=data.dtype, device=data.device).unsqueeze(1)

    # Channel dropout: randomly zero out some channels
    def channel_dropout(self, data, dropout_rate=0.1):
        batch_size, _, num_channels, _ = data.shape
        dropout_mask = torch.rand(batch_size, 1, num_channels, 1, device=data.device) > dropout_rate
        return data * dropout_mask.float()

    # Channel shuffling: randomly shuffle channel order
    def channel_shuffle(self, data):
        batch_size, _, num_channels, _ = data.shape
        shuffled_data = data.clone()
        for i in range(batch_size):
            indices = torch.randperm(num_channels)
            shuffled_data[i, :, :, :] = data[i, :, indices, :]
        return shuffled_data

    def time_dropout(self, data, dropout_rate=0.1):
        batch_size, _, num_channels, num_time_points = data.shape
        dropout_mask = torch.rand(batch_size, 1, 1, num_time_points, device=data.device) > dropout_rate
        return data * dropout_mask.float()

    # Time-series permutation: divide the time-series into 3 parts and shuffle them
    def time_series_permutation(self, data):
        batch_size, _, num_channels, num_time_points = data.shape
        permuted_data = data.clone()
        # Divide into 3 segments and shuffle
        split_points = [0, num_time_points // 3, 2 * num_time_points // 3, num_time_points]
        for i in range(batch_size):
            segments = [data[i, :, :, split_points[k]:split_points[k + 1]] for k in range(3)]
            np.random.shuffle(segments)
            permuted_data[i, :, :, :] = torch.cat(segments, dim=-1)
        return permuted_data

    # Amplitude scaling: scale the amplitude of the signals by a random factor
    def amplitude_scaling(self, data, scale_range=(0.8, 1.2)):
        batch_size, _, num_channels, _ = data.shape
        scale_factors = torch.rand(batch_size, 1, num_channels, 1, device=data.device) * (
                    scale_range[1] - scale_range[0]) + scale_range[0]
        return data * scale_factors

    def data_abs(self, data):
        return torch.abs(data)

    def signflip(self, data):
        return data * -1

    def add_noise(self, data):
        batch_size, _, num_channels, num_time_points = data.shape
        noise = torch.randn(batch_size, 1, num_channels, num_time_points // 5, device=data.device)
        data[:, :, :, -num_time_points // 5:-1] += noise
        return data

    def add_noise_P3(self, data):
        batch_size, _, num_channels, _ = data.shape
        p3_region = [38, 64]
        noise = torch.randn(batch_size, 1, num_channels, p3_region[1] - p3_region[0], device=data.device)
        data[:, :, :, p3_region[0]:p3_region[1]] += noise
        return data

    def replace_with_noise(self, data):
        batch_size, _, num_channels, num_time_points = data.shape
        noise = torch.randn(batch_size, 1, num_channels, num_time_points // 5, device=data.device)
        data[:, :, :, -num_time_points // 5:-1] = noise
        return data

    def augment(self, data):
        return [
            self.signflip(data.clone()),
            # self.replace_with_noise(data.clone()),
            # self.bandpass_filter(data.clone()),
            # self.channel_dropout(data.clone()),
            # self.channel_shuffle(data.clone()),
            # self.time_series_permutation(data.clone()),
            self.amplitude_scaling(data.clone()),
            # self.data_abs(data.clone()),
            # self.time_dropout(data.clone()),
        ]
