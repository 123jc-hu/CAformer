import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.backends import cudnn

from Deep_learning.rsvp_classification_with_CSSL import CAFormerTrainer
from project_utils.naming import (
    canonical_model_name,
    experiment_setting_name,
    fold_data_directory_name,
    infer_subject_count,
    result_excel_name,
)


def seed_torch(seed=2024):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    cudnn.benchmark = False


def setup_logging(model_name, log_tag=""):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    log_name = f"{model_name}.log" if not log_tag else f"{model_name}_{log_tag}.log"

    file_handler = logging.FileHandler(log_dir / log_name, mode="a")
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def load_yaml_config(config_path):
    if not config_path:
        return {}

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping at the top level: {config_file}")

    return config


def build_parser():
    parser = argparse.ArgumentParser(description="CAFormer RSVP classification")
    parser.add_argument("--config", type=str, default=None, help="path to a YAML config file")
    parser.add_argument("--model", type=str, default="CAFormer", help="model name; TTMTN is accepted as a legacy alias")
    parser.add_argument("--is_training", action="store_true", help="train before testing")
    parser.add_argument("--random_seed", type=int, default=2024, help="random seed")

    parser.add_argument("--dataset", type=str, default="THU", choices=["THU", "CAS", "GIST"])
    parser.add_argument("--data_variant", type=str, default="standard")
    parser.add_argument("--n_fold", type=int, default=5)
    parser.add_argument("--remove_num", type=int, default=0)
    parser.add_argument("--subject_start", type=int, default=1)
    parser.add_argument("--subject_limit", type=int, default=None)
    parser.add_argument("--result_tag", type=str, default="")
    parser.add_argument("--checkpoint_tag", type=str, default="")
    parser.add_argument("--log_tag", type=str, default="")

    parser.add_argument("--train_epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1)
    parser.add_argument("--temperature_tau", type=float, default=0.01)
    parser.add_argument("--augmentation_profile", type=str, default="dataset_default")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--use_gpu", type=bool, default=True)

    parser.add_argument("--n_channels", type=int, default=62)
    parser.add_argument("--fs", type=int, default=128)
    parser.add_argument("--n_class", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.7)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--n_heads", type=int, default=1)
    return parser


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining_args = config_parser.parse_known_args()

    parser = build_parser()
    yaml_config = load_yaml_config(config_args.config)
    valid_keys = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(yaml_config) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unsupported config keys in {config_args.config}: {unknown_keys}")

    if yaml_config:
        parser.set_defaults(**yaml_config)

    args = parser.parse_args(remaining_args)
    args.config = config_args.config
    return args


def resolve_subject_indices(total_subjects, subject_start=1, subject_limit=None):
    if subject_start < 1:
        raise ValueError("subject_start must be >= 1")
    if subject_limit is not None and subject_limit < 1:
        raise ValueError("subject_limit must be >= 1 when provided")

    start_index = subject_start - 1
    if start_index >= total_subjects:
        raise ValueError(f"subject_start={subject_start} exceeds total subjects={total_subjects}")

    end_index = total_subjects if subject_limit is None else min(total_subjects, start_index + subject_limit)
    return list(range(start_index, end_index))


def save_run_config(result_folder, args, result_setting):
    config_payload = dict(vars(args))
    config_payload["result_setting"] = result_setting
    config_path = Path(result_folder) / "run_config.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config_payload, file, sort_keys=True, allow_unicode=True)


def finalize_args(args):
    args.model_display = args.model
    args.model = canonical_model_name(args.model)
    if args.model != "CAFormer":
        raise ValueError(f"Public release only supports CAFormer, got: {args.model_display}")
    args.use_gpu = bool(torch.cuda.is_available())
    if args.dataset == "GIST":
        args.n_channels = 32
    return args


def main(args):
    time_start = time.time()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    logging.info(f"Starting run for model: {args.model_display} (canonical: {args.model})")
    if args.config:
        logging.info(f"Loaded config: {args.config}")

    total_subjects = infer_subject_count(
        args.dataset,
        args.n_fold,
        dataset_root=os.path.join(".", "Dataset"),
        data_variant=args.data_variant,
    )
    subject_indices = resolve_subject_indices(total_subjects, args.subject_start, args.subject_limit)
    subject_count = len(subject_indices)

    logging.info(
        f"Running subjects sub{subject_indices[0] + 1}-sub{subject_indices[-1] + 1} "
        f"({subject_count}/{total_subjects})"
    )

    setting = experiment_setting_name(args.n_fold, args.model, args.dataset)
    result_setting = setting if not args.result_tag else f"{setting}_{args.result_tag}"
    checkpoint_setting = setting if not args.checkpoint_tag else f"{setting}_{args.checkpoint_tag}"
    result_folder = os.path.join("./results", result_setting)
    os.makedirs(result_folder, exist_ok=True)
    save_run_config(result_folder, args, result_setting)

    sub_mean_auc = np.empty(subject_count)
    sub_mean_ba = np.empty(subject_count)
    sub_mean_f1 = np.empty(subject_count)
    sub_mean_tpr = np.empty(subject_count)
    sub_mean_tnr = np.empty(subject_count)
    result_excel_information = []
    fold_dirname = fold_data_directory_name(args.n_fold, args.data_variant)

    for output_index, sub_index in enumerate(subject_indices):
        args.sub_name = f"sub{sub_index + 1}"
        data_path = os.path.join(
            os.getcwd(),
            "Dataset",
            args.dataset,
            fold_dirname,
            f"{args.sub_name}.npz",
        )
        all_data = np.load(data_path)
        train_val_data = all_data["train_data"]
        train_val_label = all_data["train_label"]
        test_data = all_data["test_data"]
        test_label = all_data["test_label"]
        fold_length = all_data["fold_length"]

        train_val_data_list = [
            train_val_data[i * fold_length:(i + 1) * fold_length]
            for i in range(args.n_fold)
        ]
        train_val_label_list = [
            train_val_label[i * fold_length:(i + 1) * fold_length]
            for i in range(args.n_fold)
        ]
        train_val_set = (train_val_data_list, train_val_label_list)
        test_set = (test_data, test_label)

        test_auc = np.empty(args.n_fold)
        test_ba = np.empty(args.n_fold)
        test_f1 = np.empty(args.n_fold)
        test_tpr = np.empty(args.n_fold)
        test_tnr = np.empty(args.n_fold)

        result_path = os.path.join(result_folder, f"result_classification_{args.sub_name}.txt")
        with open(result_path, "w", encoding="utf-8"):
            pass

        for fold_i in range(args.n_fold):
            seed_torch(args.random_seed)
            exp = CAFormerTrainer(args)

            if args.is_training:
                print(f"Starting fold {fold_i} training for {checkpoint_setting}")
                exp.train(checkpoint_setting, fold_i, train_val_set)

            print(f"Starting fold {fold_i} testing for {checkpoint_setting}")
            test_results = exp.test(checkpoint_setting, fold_i, test_set, result_path)
            test_auc[fold_i], test_ba[fold_i], test_f1[fold_i] = test_results[:3]
            test_tpr[fold_i], test_tnr[fold_i] = test_results[4:6]
            torch.cuda.empty_cache()

        logging.info(f"{args.sub_name} {args.n_fold}fold result for {setting}")
        n_fold_mean_auc = test_auc.mean()
        n_fold_mean_ba = test_ba.mean()
        n_fold_mean_f1 = test_f1.mean()
        n_fold_mean_tpr = test_tpr.mean()
        n_fold_mean_tnr = test_tnr.mean()

        logging.info(
            f"AUC: {n_fold_mean_auc:.4f}, BA: {n_fold_mean_ba:.4f}, "
            f"F1: {n_fold_mean_f1:.4f}, TPR: {n_fold_mean_tpr:.4f}, TNR: {n_fold_mean_tnr:.4f}"
        )

        sub_mean_auc[output_index] = n_fold_mean_auc.round(4)
        sub_mean_ba[output_index] = n_fold_mean_ba.round(4)
        sub_mean_f1[output_index] = n_fold_mean_f1.round(4)
        sub_mean_tpr[output_index] = n_fold_mean_tpr.round(4)
        sub_mean_tnr[output_index] = n_fold_mean_tnr.round(4)
        result_excel_information.append([
            args.dataset,
            args.sub_name,
            n_fold_mean_auc.round(4),
            n_fold_mean_ba.round(4),
            n_fold_mean_f1.round(4),
            n_fold_mean_tpr.round(4),
            n_fold_mean_tnr.round(4),
        ])

    logging.info("")
    logging.info(f"Mean result for {args.dataset} dataset")
    logging.info(
        f"Mean AUC: {sub_mean_auc.mean():.4f}+{sub_mean_auc.std():.4f}, "
        f"Mean BA: {sub_mean_ba.mean():.4f}+{sub_mean_ba.std():.4f}, "
        f"Mean F1: {sub_mean_f1.mean():.4f}+{sub_mean_f1.std():.4f}, "
        f"Mean TPR: {sub_mean_tpr.mean():.4f}+{sub_mean_tpr.std():.4f}, "
        f"Mean TNR: {sub_mean_tnr.mean():.4f}+{sub_mean_tnr.std():.4f}"
    )
    result_excel_information.append([
        args.dataset,
        "Mean",
        f"{sub_mean_auc.mean().round(4)}+{sub_mean_auc.std().round(4)}",
        f"{sub_mean_ba.mean().round(4)}+{sub_mean_ba.std().round(4)}",
        f"{sub_mean_f1.mean().round(4)}+{sub_mean_f1.std().round(4)}",
        f"{sub_mean_tpr.mean().round(4)}+{sub_mean_tpr.std().round(4)}",
        f"{sub_mean_tnr.mean().round(4)}+{sub_mean_tnr.std().round(4)}",
    ])
    result_excel = pd.DataFrame(
        result_excel_information,
        columns=["Dataset", "Subject", "AUC", "BA", "F1-score", "TPR", "TNR"],
    )
    result_excel_path = os.path.join(result_folder, result_excel_name(args.model))
    result_excel.to_excel(result_excel_path, index=False)
    logging.info(f"Results saved in {result_excel_path}")
    logging.info(f"Execution time: {(time.time() - time_start) / 60:.2f} minutes \n")


if __name__ == "__main__":
    args = finalize_args(parse_args())
    setup_logging(args.model, args.log_tag or args.result_tag)
    main(args)
