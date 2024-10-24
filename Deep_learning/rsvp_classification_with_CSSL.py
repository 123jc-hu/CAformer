import torch
import torch.nn as nn
import numpy as np
import os
from Deep_learning.exp_basic import ExpBasic
from Data_Processing.make_dataset import dataloader
from Deep_learning.tools import EarlyStopping, cal_auc, cal_ba, cal_F1_score, cal_confusion_matrix, plot_train_val_figure, calculate_tpr_tnr
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path


class RsvpClassificationWithCSSL(ExpBasic):
    def _build_model(self):
        return self.model_dict[self.args.model].Model(self.args).float()

    def _select_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        # return torch.optim.RAdam(self.model.parameters(), lr=self.args.learning_rate,
        #                         weight_decay=self.args.weight_decay)

    def _select_criterion(self, y_data):
        target_num = len(np.where(y_data == 1)[0])
        nontarget_num = len(np.where(y_data == 0)[0])
        class_weight = torch.tensor([1, nontarget_num / target_num]).to(self.device)
        return nn.CrossEntropyLoss(weight=class_weight)

    def _augment_data(self, batch_x, augmentor):
        augmented_data = augmentor.augment(batch_x)
        _, features_aug = self.model(torch.cat(augmented_data, dim=0))
        f1, f2 = torch.split(features_aug, batch_x.size(0), dim=0)
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

    def validate(self, vali_loader, criterion, criterion_c, augmentor, gamma=0.5):
        self.model.eval()
        preds, total_outputs, total_label, total_feature_aug = [], [], [], []
        with torch.no_grad():
            for batch_x, label in vali_loader:
                batch_x, label = batch_x.to(self.device), label.to(self.device)
                outputs, _ = self.model(batch_x)
                features_aug = self._augment_data(batch_x, augmentor)
                preds.append(F.softmax(outputs, dim=-1))
                total_outputs.append(outputs)
                total_label.append(label)
                total_feature_aug.append(features_aug)

            total_outputs = torch.cat(total_outputs)
            total_label = torch.cat(total_label)
            total_feature_aug = torch.cat(total_feature_aug)

            ce_loss = criterion(total_outputs, total_label.long())
            sc_loss = criterion_c(total_feature_aug)
            loss = ce_loss + sc_loss * gamma

        preds = torch.cat(preds)
        predictions = torch.argmax(preds, dim=1).cpu().numpy()
        trues = total_label.cpu().numpy()
        auc = cal_auc(trues, preds[:, 1].cpu().numpy())
        ba = cal_ba(trues, predictions)
        tpr, tnr = calculate_tpr_tnr(trues, predictions)

        return loss.cpu().numpy(), ce_loss.cpu().numpy(), sc_loss.cpu().numpy(), auc, ba, tpr, tnr

    def train(self, setting, fold_i, train_val_set, verbose=True):
        Sup_loss = SupConLoss(temperature=0.01, base_temperature=0.01)
        augmentor = DataAugmentation()
        train_val_set_copy = copy.deepcopy(train_val_set)
        val_x, val_y = train_val_set_copy[0].pop(fold_i), train_val_set_copy[1].pop(fold_i)
        train_x = np.concatenate(train_val_set_copy[0])
        train_y = np.concatenate(train_val_set_copy[1])
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
        val_tpr_list, val_tnr_list = [], []
        train_ce_loss_list, train_c_loss_list = [], []
        val_ce_loss_list, val_c_loss_list = [], []
        gamma = 0.1
        lr_list = []

        for epoch in range(self.args.train_epochs):
            current_lr = model_optim.param_groups[0]["lr"]
            lr_list.append(current_lr)
            train_loss, train_ce_loss, train_c_loss = [], [], []

            self.model.train()
            for batch_x, label in train_loader:
                if batch_x.shape[0] < 10:
                    continue
                model_optim.zero_grad()
                batch_x, label = batch_x.to(self.device), label.to(self.device)
                outputs, _ = self.model(batch_x)
                features_aug = self._augment_data(batch_x, augmentor)
                ce_loss = criterion(outputs, label.long())
                sup_loss = Sup_loss(features_aug)
                loss = ce_loss + sup_loss * gamma
                train_loss.append(loss.item())
                train_ce_loss.append(ce_loss.item())
                train_c_loss.append(sup_loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            train_loss = np.mean(train_loss)
            train_ce_loss = np.mean(train_ce_loss)
            train_c_loss = np.mean(train_c_loss)

            val_loss, val_ce_loss, val_c_loss, val_auc, val_ba, val_tpr, val_tnr = self.validate(val_loader, criterion, Sup_loss, augmentor, gamma)

            train_loss_list.append(train_loss)
            train_ce_loss_list.append(train_ce_loss)
            train_c_loss_list.append(train_c_loss)
            val_loss_list.append(val_loss)
            val_ce_loss_list.append(val_ce_loss)
            val_c_loss_list.append(val_c_loss)
            val_auc_list.append(val_auc)
            val_ba_list.append(val_ba)
            val_tpr_list.append(val_tpr)
            val_tnr_list.append(val_tnr)

            print(f"{self.args.sub_name} Fold{fold_i+1}/{self.args.n_fold} Epoch: {epoch + 1}, learning rate: {current_lr} | Train Loss: {train_loss:.3f} Val Loss: {val_loss:.3f} Val AUC: {val_auc:.3f} Val BA: {val_ba:.3f}")
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            scheduler.step(val_loss)

        # 加载最佳模型
        self.model.load_state_dict(torch.load(early_stopping.path))

        # 画图看看结果
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
                [train_ce_loss_list, val_ce_loss_list],
                ["Train CE Loss", "Val CE Loss"],
                "Epochs",
                "CE Loss",
                os.path.join(path, f'{self.args.sub_name}_fold{fold_i + 1}_ce_loss.png')
            )
            self.plot_and_save(
                epochs_range,
                [train_c_loss_list, val_c_loss_list],
                ["Train C Loss", "Val C Loss"],
                "Epochs",
                "CSSL Loss",
                os.path.join(path, f'{self.args.sub_name}_fold{fold_i + 1}_cssl_loss.png')
            )
            self.plot_and_save(
                epochs_range,
                [val_auc_list, val_ba_list],
                ["Val AUC", "Val BA"],
                "Epochs",
                "Score",
                os.path.join(path, f'{self.args.sub_name}_fold{fold_i + 1}_auc_ba.png')
            )

            self.plot_and_save(
                epochs_range,
                [val_tpr_list, val_tnr_list],
                ["Val TPR", "Val TNR"],
                "Epochs",
                "Rate",
                os.path.join(path, f'{self.args.sub_name}_fold{fold_i + 1}_tpr_tnr.png')
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

        predictions = torch.argmax(preds, dim=1).cpu().numpy()
        trues = trues.cpu().numpy()

        auc = cal_auc(trues, preds[:, 1].cpu().numpy())
        ba = cal_ba(trues, predictions)
        f1_score = cal_F1_score(trues, predictions)
        conf_matrix = cal_confusion_matrix(trues, predictions)
        tpr, tnr = calculate_tpr_tnr(trues, predictions)

        result_str = (f'{self.args.sub_name} AUC:{auc:.4f} BA:{ba:.4f} '
                      f'conf_matrix:{conf_matrix} TPR:{tpr:.4f} TNR:{tnr:.4f}')
        print(result_str)

        with open(result_path, 'a') as f:
            f.write(f'{setting}\n')
            f.write(f'{result_str}\n\n')

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
    def amplitude_scaling(self, data, scale_range=(0.9, 1.1)):
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
            self.add_noise(data.clone()),
            # self.channel_dropout(data.clone()),
            # self.channel_shuffle(data.clone()),
            # self.time_series_permutation(data.clone()),
            # self.amplitude_scaling(data.clone()),
            # self.data_abs(data.clone()),
            # self.time_dropout(data.clone()),
        ]
