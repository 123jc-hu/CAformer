import matplotlib.pyplot as plt
import numpy as np
import os
import mne
import scipy.io as sio
from Data_Processing.data_process import DataPreprocess


data_path = '.\Dataset\THU\preprocessed\sub1A.mat'
all_data = sio.loadmat(data_path)
x_data, y_data = all_data["x_data"], all_data["y_data"]

target_data = x_data[y_data == 1]
non_target_data = x_data[y_data == 2]
target_mean = np.mean(target_data, axis=0)
target_ch1 = target_mean[1, :]
target_ch2 = target_mean[27, :]
target_ch3 = target_mean[59, :]
x = np.linspace(0, 1000, 128)
plt.plot(x, target_ch1)
plt.plot(x, target_ch2)
plt.plot(x, target_ch3)
plt.legend(['FPz', 'Cz', 'Oz'])
plt.show()

