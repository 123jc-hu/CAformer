import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from Models.ablation_study import TTMTN
import os
import argparse
from main import seed_torch
import pandas as pd
import mne

if __name__ == '__main__':
    config = {
        "font.family": 'serif',
        "font.serif": ['Times New Roman', 'Times', 'STIXGeneral'],
        "mathtext.fontset": 'stix',  # 数学公式使用 STIX，这本身就是 Times 风格
        # "axes.unicode_minus": False,  # 解决负号显示问题
        # "font.size": 12,
        # "axes.labelsize": 14,
        # "xtick.labelsize": 12,
        # "ytick.labelsize": 12,
        # "legend.fontsize": 11,
    }
    plt.rcParams.update(config)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 或者 ":16:8"

    # parameters setting
    parser = argparse.ArgumentParser(description='Salience Map Analysis')
    parser.add_argument("--n_channels", type=int, default=62, help="channels num of EEG dataset")
    parser.add_argument("--fs", type=int, default=128, help="fs of dataset")
    parser.add_argument("--n_class", type=int, default=2, help="number of classes(RSVP==2)")
    parser.add_argument("--dropout_rate", type=float, default=0.7, help="dropout rate of layers")
    parser.add_argument("--projection_dim", type=int, default=64, help="projection dim of supervised learning")
    parser.add_argument("--e_layers", type=int, default=2, help="numbers of Transformer depth")
    parser.add_argument("--d_model", type=int, default=16, help="embedding size of input")
    parser.add_argument('--dropout', type=float, default=0.7, help='dropout')
    parser.add_argument("--n_heads", type=int, default=4, help="numbers of multi-heads attention")
    args = parser.parse_args()

    # choose data from THU dataset
    dataset_name = 'THU'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TTMTN.Model(args).to(device)

    # Create a 3x2 grid for subplots
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), gridspec_kw={'wspace': 0.2, 'hspace': 0.2})  # 3 rows, 2 columns

    # Load custom sensor location
    sensor_dataframe = pd.read_excel('.\\Deep_learning\\CAT\\custom_sensor_dataframe.xlsx', index_col=0)
    channels1020 = np.array(sensor_dataframe.index)
    value1020 = np.array(sensor_dataframe)
    list1020 = dict(zip(channels1020, value1020))

    biosemi_montage = mne.channels.make_dig_montage(list1020,
                                                    nasion=[5.27205792e-18, 8.60992398e-02, -4.01487349e-02],
                                                    lpa=[-0.08609924, -0., -0.04014873],
                                                    rpa=[0.08609924, 0., -0.04014873])
    info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=128., ch_types='eeg')

    all_images = []  # For storing the images for colorbar sharing

    for i, sub_id in enumerate([1, 10, 20, 30, 40, 50]):
        seed_torch(2024)
        data_path = f'.\\Dataset\\{dataset_name}\\fold5_data\\sub{sub_id}.npz'
        all_data = np.load(data_path)
        input_data, input_label = all_data['train_data'], all_data['train_label']

        input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(dim=1).to(device)
        # print(input_data.shape)
        input_data.requires_grad_()  # Enable gradient computation for the input

        # load the model
        pre_trained_model_path = f'.\\checkpoints\\5fold_TTMTN_{dataset_name}\\sub{sub_id}_fold1_checkpoint.pth'
        model.load_state_dict(torch.load(pre_trained_model_path))

        # perform forward pass
        output, _ = model(input_data)
        target_index = np.where(input_label == 1)[0]
        non_target_index = np.where(input_label == 0)[0]

        # Backpropagate the output score
        target_class = torch.argmax(output, dim=1)
        selected_output = output[range(output.shape[0]), target_class]
        selected_output.sum().backward()

        # Calculate the gradient of the input
        target_gradients = input_data.grad[target_index].data
        non_target_gradients = input_data.grad[non_target_index].data

        # Calculate the mean gradient for each channel
        target_mean_gradient = torch.mean(target_gradients, dim=0).squeeze().cpu().numpy()
        non_target_mean_gradient = torch.mean(non_target_gradients, dim=0).squeeze().cpu().numpy()

        target_mean_ch = np.mean(target_mean_gradient, axis=1)
        target_mean_ch = (target_mean_ch - np.min(target_mean_ch)) / (np.max(target_mean_ch) - np.min(target_mean_ch)) * 2 - 1
        non_target_mean_ch = np.mean(non_target_mean_gradient, axis=1)
        non_target_mean_ch = (non_target_mean_ch - np.min(non_target_mean_ch)) / (np.max(non_target_mean_ch) - np.min(non_target_mean_ch)) * 2 - 1

        evoked1 = mne.EvokedArray(target_mean_gradient, info)
        evoked1.set_montage(biosemi_montage)
        evoked2 = mne.EvokedArray(non_target_mean_gradient, info)
        evoked2.set_montage(biosemi_montage)
        # plt.figure(1)
        # Plot target_mean_ch
        ax_target = axes[i // 2, (i % 2) * 2]  # Left subplot for target
        im, _ = mne.viz.plot_topomap(target_mean_ch, evoked1.info, show=False, axes=ax_target)
        ax_target.set_title(f"Subject {sub_id} - Target")

        # Plot non_target_mean_ch
        ax_non_target = axes[i // 2, (i % 2) * 2 + 1]  # Right subplot for non-target
        im_non_target, _ = mne.viz.plot_topomap(non_target_mean_ch, evoked2.info, show=False, axes=ax_non_target)
        ax_non_target.set_title(f"Subject {sub_id} - Non-Target")

        # Store images for colorbar
        all_images.append(im)

    # Add a shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Colorbar location
    fig.colorbar(all_images[0], cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit the colorbar
    plt.savefig('spatial_saliency_map.png', dpi=600, bbox_inches='tight')
    fig.savefig("spatial_saliency_map.pdf",
                dpi=600, bbox_inches='tight', format='pdf')
    print('the end')
