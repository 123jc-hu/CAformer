import os
import argparse
from main import seed_torch
from Data_Processing.make_dataset import load_preprocessed_data_2days, load_preprocessed_data_for_GIST
import numpy as np

if __name__ == '__main__':
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 或者 ":16:8"

    parser = argparse.ArgumentParser(description='PyTorch Data Split and Save')
    parser.add_argument('--dataset', default='GIST', type=str, help='dataset name')
    parser.add_argument("--sub_name", type=str, default="sub1", help="choose sub for each dataset")
    parser.add_argument("--n_fold", type=int, default=5, help="N fold cross-validation")
    parser.add_argument("--remove_num", type=int, default=0, help="number of removed non-targets")
    parser.add_argument("--random_seed", type=int, default=2024, help="choose random seed to help repeat")

    args = parser.parse_args()

    for sub_index in range(0, 55):
        seed_torch(args.random_seed)
        args.sub_name = f'sub{sub_index+1}'
        print(f'Processing {args.dataset}---{args.sub_name}...')

        # load data
        # train_val_set, test_set = load_preprocessed_data_2days(args)
        train_val_set, test_set = load_preprocessed_data_for_GIST(args)

        train_data = [train_val_set[0][i] for i in range(len(train_val_set[0]))]
        train_label = [train_val_set[1][i] for i in range(len(train_val_set[1]))]
        fold_length = train_label[0].shape[0]
        train_data = np.concatenate(train_data, axis=0)
        train_label = np.concatenate(train_label, axis=0)
        test_data = test_set[0]
        test_label = test_set[1]

        # save data
        save_folder = os.path.join(os.getcwd(), 'Dataset', args.dataset, 'fold5_data')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, f'{args.sub_name}.npz')
        np.savez(save_path, train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label,
                 fold_length=fold_length)

    print('finish!')
