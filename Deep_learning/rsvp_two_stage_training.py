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


class RsvpClassification_two_stage(ExpBasic):
    def __init__(self, args):
        super(RsvpClassification_two_stage, self).__init__(args)

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

    def init_prototypes(self, train_loader) -> torch.Tensor:
        """Compute class prototypes from support samples."""
        self.model.eval()
        init_target_prototypes, init_nontarget_prototypes = [], []
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(train_loader):
                batch_x = batch_x.to(self.device)
                label = label.to(self.device)
                target_label = torch.where(label == 1)[0]
                nontarget_label = torch.where(label == 0)[0]
                _, features = self.model(batch_x)
                target_features = features[target_label]  # (NsT, D)
                nontarget_features = features[nontarget_label]   # (NsNT, D)
                init_target_prototypes.append(target_features)
                init_nontarget_prototypes.append(nontarget_features)
        init_target_prototypes = torch.cat(init_target_prototypes, dim=0)
        init_nontarget_prototypes = torch.cat(init_nontarget_prototypes, dim=0)
        init_target_prototypes = torch.mean(init_target_prototypes, dim=0, keepdim=True)
        init_nontarget_prototypes = torch.mean(init_nontarget_prototypes, dim=0, keepdim=True)
        class_prototypes = torch.cat((init_target_prototypes, init_nontarget_prototypes), dim=0)
        return class_prototypes

    def psc_loss_binary(self, features, labels, prototype, temperature=0.1):
        # 计算相似度 (dot product) 然后除以温度参数 tau
        similarity = torch.matmul(features, prototype.T) / temperature

        # 获取正确类别的相似度
        correct_class_sim = similarity[torch.arange(len(labels)), labels]

        # 获取错误类别的相似度
        incorrect_class_sim = similarity[torch.arange(len(labels)), 1 - labels]

        # 计算分子部分
        numerator = torch.exp(correct_class_sim)

        # 计算分母部分
        denominator = numerator + torch.exp(incorrect_class_sim)

        # 计算损失
        loss = -torch.log(numerator / denominator)

        return loss.mean()

    def validate(self, vali_loader, criterion, criterion_sc, augmentor, stage):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(vali_loader):
                batch_x = batch_x.to(self.device)
                label = label.to(self.device)

                # 数据增强
                batch_x_aug = augmentor.augment(batch_x)   # list of augmented data [list1, list2]
                outputs_aug, features_aug = self.model(torch.cat([batch_x_aug[0], batch_x_aug[1]], dim=0))
                batch_size = batch_x.shape[0]
                f1, f2 = torch.split(features_aug, [batch_size, batch_size], dim=0)
                features_aug = torch.cat((f1.unsqueeze(1), f2.unsqueeze(1)), dim=1)

                outputs, features = self.model(batch_x)

                pred = outputs.detach()
                features_aug = features_aug.detach()
                ce_loss = criterion(pred, label.long())
                sup_loss = criterion_sc(features_aug, label.long())
                loss = sup_loss if stage == 1 else ce_loss
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
        Sup_loss = SupConLoss()
        augmentor = DataAugmentation()
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
        scheduler = ReduceLROnPlateau(model_optim, 'min', factor=0.5, patience=5, verbose=True)

        train_loss_list, val_loss_list, val_auc_list, val_ba_list = [], [], [], []

        if stage == 1:
            # stage 1: representation learning
            for param in self.model.ClassifierBlock.parameters():
                param.requires_grad = False
        elif stage == 2:
            # stage 2: classifier learning
            for param in self.model.ClassifierBlock.parameters():
                param.requires_grad = True
            # for param in self.model.projection_head.parameters():
            #     param.requires_grad = False
        else:
            raise ValueError("Invalid stage number!")

        for epoch in range(self.args.train_epochs):
            current_lr = model_optim.param_groups[0]["lr"]
            train_loss = []
            train_ce_loss, train_sc_loss = [], []

            self.model.train()

            for i, (batch_x, label) in enumerate(train_loader):
                gamma = 1 - epoch / 50 if epoch < 50 else 0
                model_optim.zero_grad()

                batch_x = batch_x.to(self.device)
                label = label.to(self.device)
                target_label = torch.where(label == 1)[0]
                if len(target_label) == 0:
                    continue

                # 数据增强
                batch_x_aug = augmentor.augment(batch_x)   # list of augmented data [list1, list2]
                outputs_aug, features_aug = self.model(torch.cat([batch_x_aug[0], batch_x_aug[1]], dim=0))
                batch_size = batch_x.shape[0]
                f1, f2 = torch.split(features_aug, [batch_size, batch_size], dim=0)
                features_aug = torch.cat((f1.unsqueeze(1), f2.unsqueeze(1)), dim=1)
                outputs, _ = self.model(batch_x)
                ce_loss = criterion(outputs, label.long())
                sup_loss = Sup_loss(features_aug, label.long())
                loss = sup_loss if stage == 1 else ce_loss
                train_loss.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            train_loss = torch.tensor(train_loss).mean()
            val_loss, val_auc, val_ba = self.validate(val_loader, criterion, Sup_loss, augmentor, stage)
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
        # _, test_set = load_preprocessed_data(self.args)
        test_loader = dataloader(test_set[0], test_set[1], batch_size=self.args.batch_size)
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints, setting, f'fold{fold_i+1}_checkpoint.pth')))

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
        """
        Perform sign flip on the time domain of the data.
        """
        return data * -1

    def add_noise(self, data):
        """
        Add Gaussian noise to the last 1/4 of the time domain.
        """
        batch_size, _, num_channels, num_time_points = data.shape
        noise = torch.randn(batch_size, 1, num_channels, num_time_points // 5, device=data.device)
        augmented_data = data.clone()
        augmented_data[:, :, :, -num_time_points // 5:-1] += noise
        return augmented_data

    def add_noise_P3(self, data):
        """
        Add Gaussian noise to the last 1/5 of the time domain.
        """
        p3_region = [38, 64]
        batch_size, _, num_channels, num_time_points = data.shape
        noise = torch.randn(batch_size, 1, num_channels, 64-38, device=data.device)
        augmented_data = data.clone()
        augmented_data[:, :, :, 38:64] += noise
        return augmented_data

    def replace_with_noise(self, data):
        """
        Replace the last 1/4 of the time domain with Gaussian noise.
        """
        batch_size, _, num_channels, num_time_points = data.shape
        noise = torch.randn(batch_size, 1, num_channels, num_time_points // 4, device=data.device)
        augmented_data = data.clone()
        augmented_data[:, :, :, -num_time_points // 4:] = noise
        return augmented_data

    def shuffle_channels(self, data):
        """
        Randomly shuffle the channels of the data.
        """
        batch_size, _, num_channels, num_time_points = data.shape
        shuffled_data = data.clone()
        for i in range(batch_size):
            shuffled_data[i, 0] = shuffled_data[i, 0][torch.randperm(num_channels)]
        return shuffled_data

    def augment(self, data):
        """
        Apply the augmentations and return a list of augmented data.
        """
        signflipped_data = self.signflip(data)
        noise_added_data = self.add_noise(data)
        noise_replaced_data = self.replace_with_noise(data)
        shuffled_data = self.shuffle_channels(data)
        noise_added_p3 = self.add_noise_P3(data)

        return [signflipped_data, noise_added_data]
