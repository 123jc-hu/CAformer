import random
import torch
import numpy as np
from torch.backends import cudnn
from Deep_learning.rsvp_classification import RsvpClassification
# from Deep_learning.rsvp_classification_with_sc import RsvpClassification_with_sc
from Deep_learning.rsvp_classification_for_MTCN import RsvpClassification_for_MTCN
from Deep_learning.rsvp_classification_with_CSSL import RsvpClassificationWithCSSL
import time
import os
import argparse
import logging
import pandas as pd


# 随机数种子设定
def seed_torch(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.use_deterministic_algorithms(True)
    cudnn.benchmark = False


def setup_logging(model_name):
    # 创建logger对象
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建一个处理器用于输出日志到文件
    file_handler = logging.FileHandler(f'{model_name}.log', mode='a')
    file_handler.setLevel(logging.INFO)

    # 创建一个处理器用于输出日志到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def main(args):
    # seed_torch(args.random_seed)

    for n_layers in range(1):

        time_start = time.time()
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        logging.info(f'Starting training for model: {args.model}')
        sub_num_for_test = 55
        sub_mean_auc, sub_mean_ba, sub_mean_f1 = np.empty(sub_num_for_test), np.empty(sub_num_for_test), np.empty(
            sub_num_for_test)
        sub_mean_tpr, sub_mean_tnr = np.empty(sub_num_for_test), np.empty(sub_num_for_test)

        setting = f'{args.n_fold}fold_{args.model}_{args.dataset}'
        result_folder = os.path.join('./results/', setting)
        os.makedirs(result_folder, exist_ok=True)
        result_excel_information = []

        for sub_index in range(sub_num_for_test):
            args.sub_name = f"sub{sub_index+1}"
            data_path = os.path.join(os.getcwd(), 'Dataset', args.dataset, f'fold{args.n_fold}_data',
                                     f'{args.sub_name}.npz')
            # data_path = os.path.join(os.getcwd(), 'Dataset', args.dataset, f'fold{args.n_fold}_data_for_MTCN',
            #                          f'{args.sub_name}.npz')
            all_data = np.load(data_path)
            train_val_data, train_val_label = all_data['train_data'], all_data['train_label']
            test_data, test_label = all_data['test_data'], all_data['test_label']
            fold_length = all_data['fold_length']
            train_val_data_list = [train_val_data[i * fold_length:(i + 1) * fold_length] for i in range(args.n_fold)]
            train_val_label_list = [train_val_label[i * fold_length:(i + 1) * fold_length] for i in range(args.n_fold)]
            train_val_set = (train_val_data_list, train_val_label_list)
            test_set = (test_data, test_label)
            if args.model == "MTCN":
                train_val_data_mtr, train_val_label_mtr = all_data['train_data_mtr'], all_data['train_label_mtr']
                train_val_data_msr, train_val_label_msr = all_data['train_data_msr'], all_data['train_label_msr']
                train_val_data_mtr_list = [train_val_data_mtr[9 * i * fold_length:9 * (i + 1) * fold_length] for i in
                                           range(args.n_fold)]
                train_val_label_mtr_list = [train_val_label_mtr[9 * i * fold_length:9 * (i + 1) * fold_length] for i in
                                            range(args.n_fold)]
                train_val_data_msr_list = [train_val_data_msr[8 * i * fold_length:8 * (i + 1) * fold_length] for i in
                                           range(args.n_fold)]
                train_val_label_msr_list = [train_val_label_msr[8 * i * fold_length:8 * (i + 1) * fold_length] for i in
                                            range(args.n_fold)]
                train_val_mtr_set = (train_val_data_mtr_list, train_val_label_mtr_list)
                train_val_msr_set = (train_val_data_msr_list, train_val_label_msr_list)

            test_auc, test_ba, test_f1, test_conf_matrix = np.empty(args.n_fold), np.empty(args.n_fold), np.empty(
                args.n_fold), []
            test_tpr, test_tnr = np.empty(args.n_fold), np.empty(args.n_fold)

            result_path = os.path.join(result_folder, f'result_classification_{args.sub_name}.txt')
            with open(result_path, 'w'):
                pass

            for fold_i in range(args.n_fold):
                seed_torch(args.random_seed)
                # exp = RsvpClassification_with_sc(args)
                exp = RsvpClassificationWithCSSL(args)
                # exp = RsvpClassification(args)
                # exp = RsvpClassification_for_MTCN(args)

                if args.is_training:
                    print(f'Starting fold {fold_i} training for {setting}')
                    if args.model == "MTCN":
                        exp.train(setting, fold_i, train_val_set, train_val_mtr_set, train_val_msr_set)
                    else:
                        exp.train(setting, fold_i, train_val_set)

                print(f'Starting fold {fold_i} testing for {setting}')
                test_results = exp.test(setting, fold_i, test_set, result_path)
                test_auc[fold_i], test_ba[fold_i], test_f1[fold_i] = test_results[:3]
                test_conf_matrix.append(test_results[3])
                test_tpr[fold_i], test_tnr[fold_i] = test_results[4:6]

                torch.cuda.empty_cache()

            logging.info(f'{args.sub_name} {args.n_fold}fold result for {setting}')
            N_fold_mean_AUC, N_fold_mean_BA = test_auc.mean(), test_ba.mean()
            N_fold_mean_F1, N_fold_mean_tpr, N_fold_mean_tnr = test_f1.mean(), test_tpr.mean(), test_tnr.mean()

            logging.info(f'AUC: {N_fold_mean_AUC:.4f}, BA: {N_fold_mean_BA:.4f}, '
                         f'F1: {N_fold_mean_F1:.4f}, TPR: {N_fold_mean_tpr:.4f}, TNR: {N_fold_mean_tnr:.4f}')

            sub_mean_auc[sub_index] = N_fold_mean_AUC.round(4)
            sub_mean_ba[sub_index] = N_fold_mean_BA.round(4)
            sub_mean_f1[sub_index] = N_fold_mean_F1.round(4)
            sub_mean_tpr[sub_index] = N_fold_mean_tpr.round(4)
            sub_mean_tnr[sub_index] = N_fold_mean_tnr.round(4)
            result_excel_information.append([args.dataset, args.sub_name, N_fold_mean_AUC.round(4), N_fold_mean_BA.round(4),
                                             N_fold_mean_F1.round(4), N_fold_mean_tpr.round(4), N_fold_mean_tnr.round(4)])

        logging.info('')
        logging.info(f'Mean result for {args.dataset} dataset')
        logging.info(f'Mean AUC: {sub_mean_auc.mean():.4f}+{sub_mean_auc.std():.4f}, Mean BA: {sub_mean_ba.mean():.4f}+{sub_mean_ba.std():.4f}, '
                     f'Mean F1: {sub_mean_f1.mean():.4f}+{sub_mean_f1.std():.4f}, Mean TPR: {sub_mean_tpr.mean():.4f}+{sub_mean_tpr.std():.4f}, '
                     f'Mean TNR: {sub_mean_tnr.mean():.4f}+{sub_mean_tnr.std():.4f}')
        result_excel_information.append([args.dataset, 'Mean',
                                         f'{sub_mean_auc.mean().round(4)}+{sub_mean_auc.std().round(4)}',
                                         f'{sub_mean_ba.mean().round(4)}+{sub_mean_ba.std().round(4)}',
                                         f'{sub_mean_f1.mean().round(4)}+{sub_mean_f1.std().round(4)}',
                                         f'{sub_mean_tpr.mean().round(4)}+{sub_mean_tpr.std().round(4)}',
                                         f'{sub_mean_tnr.mean().round(4)}+{sub_mean_tnr.std().round(4)}'])
        result_excel = pd.DataFrame(result_excel_information,
                                    columns=["Dataset", "Subject", "AUC", "BA", "F1-score", "TPR", "TNR"])
        result_excel_path = os.path.join(result_folder, f'result_classification_{args.model}.xlsx')
        result_excel.to_excel(result_excel_path, index=False)
        logging.info(f"Results saved in {result_excel_path}")

        logging.info(f'Execution time: {(time.time() - time_start) / 60:.2f} minutes \n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Learning Models")
    parser.add_argument("--model", type=str, required=True, default="TTMTN", help="model name")
    parser.add_argument("--is_training", action="store_true", help="train or just test")
    parser.add_argument("--random_seed", type=int, default=2024, help="random seed")

    parser.add_argument("--dataset", type=str, default="THU", help="dataset name")
    parser.add_argument("--n_fold", type=int, default=5, help="N fold cross-validation")
    parser.add_argument("--remove_num", type=int, default=0, help="number of removed non-targets")

    parser.add_argument("--train_epochs", type=int, default=400, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="set batch size per epoch")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="optimizer weight decay")
    parser.add_argument("--patience", type=int, default=20,
                        help="How long to wait after last time validation loss improved")
    parser.add_argument("--use_gpu", type=bool, default=True, help="use GPU")

    # 网络参数
    parser.add_argument("--n_channels", type=int, default=62, help="channels num of EEG dataset")
    parser.add_argument("--fs", type=int, default=128, help="fs of dataset")
    parser.add_argument("--n_class", type=int, default=2, help="number of classes(RSVP==2)")
    parser.add_argument("--dropout_rate", type=float, default=0.7, help="dropout rate of layers")
    parser.add_argument("--projection_dim", type=int, default=64, help="projection dim of supervised learning")
    # Transformer参数
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument("--e_layers", type=int, default=1, help="numbers of Transformer depth")
    parser.add_argument("--d_model", type=int, default=16, help="embedding size of input")
    parser.add_argument('--dropout', type=float, default=0.7, help='dropout')
    parser.add_argument("--n_heads", type=int, default=1, help="numbers of multi-heads attention")

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False
    if args.dataset == "GIST":
        args.n_channels = 32

    setup_logging(args.model)
    main(args)
