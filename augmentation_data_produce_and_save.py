import os
import numpy as np


# Channel dropout
def channel_dropout(data, dropout_rate=0.1):
    batch_size, num_channels, num_time_points = data.shape
    dropout_mask = np.random.rand(batch_size, num_channels) > dropout_rate
    return data * dropout_mask[:, :, np.newaxis]


# Channel shuffling
def channel_shuffle(data):
    batch_size, num_channels, num_time_points = data.shape
    shuffled_data = data.copy()
    for i in range(batch_size):
        indices = np.random.permutation(num_channels)
        shuffled_data[i, :, :] = data[i, indices, :]
    return shuffled_data


# Time dropout
def time_dropout(data, dropout_rate=0.1):
    batch_size, num_channels, num_time_points = data.shape
    dropout_mask = np.random.rand(batch_size, num_time_points) > dropout_rate
    return data * dropout_mask[:, np.newaxis, :]


# Time-series permutation
def time_series_permutation(data):
    batch_size, num_channels, num_time_points = data.shape
    permuted_data = data.copy()
    split_points = [0, num_time_points // 3, 2 * num_time_points // 3, num_time_points]
    for i in range(batch_size):
        segments = [data[i, :, split_points[k]:split_points[k + 1]] for k in range(3)]
        np.random.shuffle(segments)
        permuted_data[i, :, :] = np.concatenate(segments, axis=-1)
    return permuted_data


# Amplitude scaling
def amplitude_scaling(data, scale_range=(0.8, 1.2)):
    batch_size, num_channels, num_time_points = data.shape
    scale_factors = np.random.uniform(scale_range[0], scale_range[1], (batch_size, num_channels, 1))
    return data * scale_factors


# Sign flip
def signflip(data):
    return data * -1


# Add noise
def add_noise(data):
    batch_size, num_channels, num_time_points = data.shape
    noise = np.random.randn(batch_size, num_channels, num_time_points // 5)
    data[:, :, -num_time_points // 5:-1] += noise
    return data


# Add noise in P3 region
def add_noise_P3(data):
    batch_size, num_channels, num_time_points = data.shape
    p3_region = [38, 64]
    noise = np.random.randn(batch_size, num_channels, p3_region[1] - p3_region[0])
    data[:, :, p3_region[0]:p3_region[1]] += noise
    return data


# Replace with noise
def replace_with_noise(data):
    batch_size, num_channels, num_time_points = data.shape
    noise = np.random.randn(batch_size, num_channels, num_time_points // 5)
    data[:, :, -num_time_points // 5:-1] = noise
    return data


# Augmentations function list
def augment(data):
    return [
        signflip(data.copy()),
        replace_with_noise(data.copy()),
        channel_dropout(data.copy()),
        amplitude_scaling(data.copy())
    ]


if __name__ == '__main__':
    dataset = 'GIST'
    for sub_id in range(55):
        # random seed
        np.random.seed(2024)
        data_path = os.path.join(os.getcwd(), 'Dataset', dataset, f'fold5_data', f'sub{sub_id+1}.npz')
        all_data = np.load(data_path)
        train_val_data, train_val_label = all_data['train_data'], all_data['train_label']
        train_val_augmented_data = augment(train_val_data)

        # save augmented data
        save_folder = os.path.join(os.getcwd(), 'Dataset', dataset, 'fold5_data', 'augmented_data')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, f'sub{sub_id+1}.npz')
        np.savez(save_path, signflip=train_val_augmented_data[0], replace_with_noise=train_val_augmented_data[1],
                 channel_dropout=train_val_augmented_data[2], amplitude_scaling=train_val_augmented_data[3])
        print(f'sub{sub_id+1} augmented data saved!')

    print('finish!')
