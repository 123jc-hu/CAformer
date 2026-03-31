import pandas as pd

if __name__ == '__main__':
    dataset_name = ['THU', 'CAS', 'GIST']
    model_name = ['DeepConvNet', 'EEGNet', 'PLNet', 'EEGInception', 'PPNN', 'EEG_Conformer', 'MTCN', 'TTMTN']
    excel_total_result = []
    for dataset in dataset_name:
        for model in model_name:
            result_path = f'.\\results\\5fold_{model}_{dataset}\\result_classification_{model}.xlsx'
            df = pd.read_excel(result_path)
            result = df.loc[df.shape[0]-1, ['AUC', 'BA', 'F1-score']].values
            auc_mean, auc_std = result[0].split('+')
            ba_mean, ba_std = result[1].split('+')
            f1_mean, f1_std = result[2].split('+')
            excel_total_result.append([dataset, model, float(auc_mean), float(auc_std), float(ba_mean), float(ba_std), float(f1_mean), float(f1_std)])
    new_df = pd.DataFrame(excel_total_result, columns=['Dataset', 'Model', 'AUC_mean', 'AUC_std', 'BA_mean', 'BA_std', 'F1_mean', 'F1_std'])
    new_df.to_excel('.\\results\\total_result.xlsx', index=False)
    print('Total result saved to .\\results\\total_result.xlsx')