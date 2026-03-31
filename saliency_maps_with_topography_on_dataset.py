import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
from Models.ablation_study import TTMTN
import os
import argparse
from main import seed_torch

if __name__ == '__main__':
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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

    dataset_dict = {'THU': 64, 'CAS': 14, 'GIST': 55}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the figure with 3 rows and 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(6, 12))

    for idx, dataset_name in enumerate(dataset_dict):
        num_subjects = dataset_dict[dataset_name]

        # initialize the model
        if dataset_name == 'GIST':
            args.n_channels = 32
        model = TTMTN.Model(args).to(device)

        # Initialize accumulators for gradients
        accumulated_target_gradients = 0
        accumulated_non_target_gradients = 0

        # Load custom sensor location
        if dataset_name != 'GIST':
            sensor_dataframe = pd.read_excel('.\\Deep_learning\\CAT\\sensor_dataframe.xlsx', index_col=0)
            channels1020 = np.array(sensor_dataframe.index)
            value1020 = np.array(sensor_dataframe)
            list1020 = dict(zip(channels1020, value1020))

            biosemi_montage = mne.channels.make_dig_montage(list1020,
                                                            nasion=[5.27205792e-18, 8.60992398e-02, -4.01487349e-02],
                                                            lpa=[-0.08609924, -0., -0.04014873],
                                                            rpa=[0.08609924, 0., -0.04014873])
        else:
            biosemi_montage = mne.channels.make_standard_montage('biosemi32')

        info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=128., ch_types='eeg')

        for sub_id in range(1, num_subjects + 1):
            seed_torch(2024)
            data_path = f'.\\Dataset\\{dataset_name}\\fold5_data\\sub{sub_id}.npz'
            all_data = np.load(data_path)
            input_data, input_label = all_data['train_data'], all_data['train_label']

            input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(dim=1).to(device)
            input_data.requires_grad_()  # Enable gradient computation for the input

            # Load the model
            pre_trained_model_path = f'.\\checkpoints\\5fold_TTMTN_{dataset_name}\\sub{sub_id}_fold1_checkpoint.pth'
            model.load_state_dict(torch.load(pre_trained_model_path))

            # Forward pass
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

            # Take the mean gradient for each subject
            target_mean_gradient = torch.mean(target_gradients, dim=0).squeeze().cpu().numpy()
            non_target_mean_gradient = torch.mean(non_target_gradients, dim=0).squeeze().cpu().numpy()

            # Accumulate gradients for averaging later
            if sub_id == 1:
                accumulated_target_gradients = target_mean_gradient
                accumulated_non_target_gradients = non_target_mean_gradient
            else:
                accumulated_target_gradients += target_mean_gradient
                accumulated_non_target_gradients += non_target_mean_gradient

        # Compute the final averaged gradients across all subjects
        final_target_mean_gradient = accumulated_target_gradients / num_subjects
        final_non_target_mean_gradient = accumulated_non_target_gradients / num_subjects

        # Average across time axis (axis=1)
        final_target_mean_ch = np.mean(final_target_mean_gradient, axis=1)
        final_non_target_mean_ch = np.mean(final_non_target_mean_gradient, axis=1)

        # Normalize the results to [-1, 1]
        final_target_mean_ch = (final_target_mean_ch - np.min(final_target_mean_ch)) / (np.max(final_target_mean_ch) - np.min(final_target_mean_ch)) * 2 - 1
        final_non_target_mean_ch = (final_non_target_mean_ch - np.min(final_non_target_mean_ch)) / (np.max(final_non_target_mean_ch) - np.min(final_non_target_mean_ch)) * 2 - 1

        evoked_target = mne.EvokedArray(final_target_mean_gradient, info)
        evoked_target.set_montage(biosemi_montage)

        evoked_non_target = mne.EvokedArray(final_non_target_mean_gradient, info)
        evoked_non_target.set_montage(biosemi_montage)

        # Plot target_mean_ch
        im, _ = mne.viz.plot_topomap(final_target_mean_ch, evoked_target.info, show=False, axes=axes[idx, 0])
        axes[idx, 0].set_title("Target Mean Gradient")

        # Plot non_target_mean_ch
        im_non_target, _ = mne.viz.plot_topomap(final_non_target_mean_ch, evoked_non_target.info, show=False, axes=axes[idx, 1])
        axes[idx, 1].set_title("Non-Target Mean Gradient")

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Colorbar location
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig('three_dataset_spatial_saliency_map.png', dpi=300, bbox_inches='tight')
    plt.show()
