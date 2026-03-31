import mne

# 假设你的文件名是 example.set（同目录下应有 example.fdt）
raw = mne.io.read_raw_eeglab(r'F:\文献汇总\数据集\GIST\Won2022_BIDS\sub-001\eeg\sub-001_task-RSVPtask_run-3_eeg.set', preload=True)

# 查看基本信息
print(raw.info)
print(raw.ch_names)
print(raw.get_data().shape)  # shape: (n_channels, n_times)

# 获取 NumPy 格式的数据
eeg_data = raw.get_data()  # 返回二维数组 (通道数, 时间点数)

events, event_id = mne.events_from_annotations(raw)
print(events[:5])  # 每行是 [时间点, 0, 事件ID]
print(event_id)    # 事件标签到整数ID的映射

# 假设某个事件名是 'stimulus'
epochs = mne.Epochs(raw, events, event_id=event_id,
                    tmin=-0.2, tmax=0.8, baseline=(None, 0),
                    preload=True)

# 访问 epochs 数据
data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)