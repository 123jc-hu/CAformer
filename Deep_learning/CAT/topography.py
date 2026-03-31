# plot EEG topograpy with mne
# https://mne.tools/stable/index.html

import mne
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import argparse
from Data_Processing.make_dataset import load_preprocessed_data_2days
import os


# parser = argparse.ArgumentParser(description="Deep Learning Models")
# parser.add_argument("--random_seed", type=int, default=2024, help="choose random seed to help repeat")
# # data loader
# parser.add_argument("--dataset", type=str, default="THU", help="choose dataset name")
# parser.add_argument("--sub_name", type=str, default="sub1", help="choose sub for each dataset")
# parser.add_argument("--remove_num", type=int, default=0, help="number of removed non-targets")
#
# # 训练参数
# parser.add_argument("--n_fold", type=int, default=5, help="N fold cross-validation")
# parser.add_argument("--batch_size", type=int, default=32, help="set batch size per epoch")
#
# # GPU
# parser.add_argument("--use_gpu", type=bool, default=True, help="use cuda or cpu")
# args = parser.parse_args()
# # load data  train_data - (trials, channels, samples)  train_label -  (label, 1)
# # current_directory = os.path.dirname(os.path.abspath(__file__))
# # print(current_directory)
# train_val_set, test_set = load_preprocessed_data_2days(args)
# train_data, train_label = np.concatenate(train_val_set[0], axis=0), np.concatenate(train_val_set[1], axis=0)
# # test_data, test_label = test_set[0], test_set[1]
#
# idx = np.where(train_label == 1)
# data_draw = train_data[idx]  # (trials, channels, samples)
# # data_mean = np.mean(data_draw[0, 0])  # 这里的数据已经是对通道来说均值为0方差为1了
# # data_std = np.std(data_draw[0, 0])
# mean_trial = np.mean(data_draw, axis=0)  # mean trial
# # use standardization or normalization to adjust
# mean_trial = (mean_trial - np.mean(mean_trial)) / np.std(mean_trial)
#
#
# mean_ch = np.mean(mean_trial, axis=1)  # mean samples with channel dimension left


# Draw topography
biosemi_montage = mne.channels.make_standard_montage('biosemi32')  # set a montage, see mne document
# biosemi_montage.plot(show_names=True)
print(biosemi_montage.ch_names)
print(biosemi_montage.get_positions())
sensor_data = biosemi_montage.get_positions()['ch_pos']
print(sensor_data)
sensor_dataframe = pd.DataFrame(sensor_data).T
print(sensor_dataframe)
sensor_dataframe.to_excel('sensor_dataframe.xlsx')

# read custom location file 更改电极位置，以匹配数据
# channels_information_path = 'D:\\learning_softwares\\Collaborative_RSVP\\Dataset\\CAS\\62-channels.loc'
# biosemi_custom_montage = mne.channels.read_custom_montage(channels_information_path)
# print(biosemi_custom_montage.ch_names)
# custom_sensor_data = biosemi_custom_montage.get_positions()['ch_pos']
# custom_sensor_dataframe = pd.DataFrame(custom_sensor_data).T
# custom_sensor_dataframe.to_excel('custom_sensor_dataframe.xlsx')
# index_list = [biosemi_montage.ch_names.index(i)  for i in biosemi_custom_montage.ch_names if i in biosemi_montage.ch_names]
# biosemi_montage.ch_names = [biosemi_montage.ch_names[i] for i in index_list]
# biosemi_montage.dig = [biosemi_montage.dig[i+3] for i in index_list]
# sensor_data = biosemi_montage.get_positions()['ch_pos']
# sensor_dataframe = pd.DataFrame(sensor_data).T
# sensor_dataframe.to_excel('sensor_dataframe.xlsx')

# 使用新的电极位置 'sensor_dataframe.xlsx'
sensor_dataframe = pd.read_excel('sensor_dataframe.xlsx', index_col=0)
channels1020 = np.array(sensor_dataframe.index)
value1020 = np.array(sensor_dataframe)
list1020 = dict(zip(channels1020, value1020))
print(list1020)
biosemi_montage = mne.channels.make_dig_montage(list1020,
                                                nasion=[5.27205792e-18, 8.60992398e-02, -4.01487349e-02],
                                                lpa=[-0.08609924, -0., -0.04014873],
                                                rpa=[0.08609924, 0., -0.04014873]
                                                )
biosemi_montage.plot(show_names=True)
plt.show()
info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=128., ch_types='eeg')  # sample rate

evoked1 = mne.EvokedArray(mean_trial, info)
evoked1.set_montage(biosemi_montage)
plt.figure(1)
# im, cn = mne.viz.plot_topomap(np.mean(mean_trial, axis=1), evoked1.info, show=False)
im, cn = mne.viz.plot_topomap(mean_ch, evoked1.info, show=False, size=5, names=evoked1.ch_names)
plt.colorbar(im)

plt.savefig('test.png')
print('the end')

