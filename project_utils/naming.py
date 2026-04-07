import os


LEGACY_MODEL_NAME_MAP = {
    "TTMTN": "CAFormer",
}

DEFAULT_SUBJECT_COUNT = {
    "THU": 64,
    "CAS": 14,
    "GIST": 55,
}


def canonical_model_name(model_name):
    return LEGACY_MODEL_NAME_MAP.get(model_name, model_name)


def experiment_setting_name(n_fold, model_name, dataset_name):
    return f"{n_fold}fold_{canonical_model_name(model_name)}_{dataset_name}"


def result_excel_name(model_name):
    return f"result_classification_{canonical_model_name(model_name)}.xlsx"


def fold_data_directory_name(n_fold, data_variant="standard"):
    if data_variant == "standard":
        return f"fold{n_fold}_data"
    return f"fold{n_fold}_data_{data_variant}"


def infer_subject_count(dataset_name, n_fold, dataset_root=None, data_variant="standard"):
    dataset_root = dataset_root or os.path.join(".", "Dataset")
    fold_dir = os.path.join(dataset_root, dataset_name, fold_data_directory_name(n_fold, data_variant))
    if os.path.isdir(fold_dir):
        subject_files = [
            file_name for file_name in os.listdir(fold_dir)
            if file_name.endswith(".npz") and file_name.startswith("sub")
        ]
        if subject_files:
            return len(subject_files)
    return DEFAULT_SUBJECT_COUNT.get(dataset_name, 0)
