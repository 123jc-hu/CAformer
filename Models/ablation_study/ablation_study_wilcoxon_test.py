from scipy.stats import wilcoxon
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd
import re

if __name__ == '__main__':
    # Initialize dataset info
    dataset_name = ['THU', 'CAS', 'GIST']
    target_model = 'TTMTN'
    compared_model = ['TTMTN_T_C', 'TTMTN_C', 'TTMTN_T']

    # Define result file path
    wilcoxon_excel_path = '..\\..\\results\\ablation_study_wilcoxon_result.xlsx'
    Path('.\\..\\results').mkdir(parents=True, exist_ok=True)

    result_data = []
    for dataset_name in dataset_name:

        # Progress bar for compared models
        with tqdm(total=len(compared_model), desc=f'Dataset: {dataset_name}', leave=True, ncols=100, unit_scale=True) as pbar:
            for model_name in compared_model:

                # load subjects results
                target_model_result_path = (f'..\\..\\results\\5fold_{target_model}_{dataset_name}\\result_classification_'
                                            f'{target_model}.xlsx')
                compared_model_result_path = (f'..\\..\\results\\5fold_{model_name}_{dataset_name}\\result_classification_'
                                              f'{model_name}.xlsx')
                target_df = pd.read_excel(target_model_result_path)
                compared_df = pd.read_excel(compared_model_result_path)
                target_model_auc, compared_model_auc = target_df['AUC'].tolist()[:-1], compared_df['AUC'].tolist()[:-1]
                target_model_ba, compared_model_ba = target_df['BA'].tolist()[:-1], compared_df['BA'].tolist()[:-1]
                target_model_f1, compared_model_f1 = target_df['F1-score'].tolist()[:-1], compared_df['F1-score'].tolist()[:-1]

                # Wilcoxon signed-rank test
                if target_model_ba and compared_model_ba:
                    ba_stat, ba_p_value = wilcoxon(target_model_ba, compared_model_ba, alternative='two-sided')
                    auc_stat, auc_p_value = wilcoxon(target_model_auc, compared_model_auc, alternative='two-sided')
                    f1_stat, f1_p_value = wilcoxon(target_model_f1, compared_model_f1, alternative='two-sided')
                    print(f'{target_model} vs {model_name}: auc_stat: {auc_stat} | auc_p-value: {auc_p_value} | '
                          f'ba_stat: {ba_stat} | ba_p-value: {ba_p_value} | '
                          f'f1_stat: {f1_stat} | f1_p-value: {f1_p_value}')

                    # Append result to the list for Excel
                    result_data.append([dataset_name, target_model, model_name, auc_stat, auc_p_value,
                                        ba_stat, ba_p_value, f1_stat, f1_p_value])

                pbar.update(1)

    # Convert results to a pandas DataFrame
    df = pd.DataFrame(result_data, columns=['dataset_name', 'target_model', 'compared_model', 'auc_stat', 'auc_p-value',
                                            'ba_stat', 'ba_p-value', 'f1_stat', 'f1_p-value'])

    # Save the DataFrame to an Excel file
    df.to_excel(wilcoxon_excel_path, index=False)
    print(f'Results saved to {wilcoxon_excel_path}')
