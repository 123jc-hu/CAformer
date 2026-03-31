import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from Models.ablation_study import TTMTN
import os
import argparse
from main import seed_torch

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
    fig, axes = plt.subplots(3, 2, figsize=(10, 6))  # 3 rows, 2 columns

    for i, sub_id in enumerate([1, 10, 20, 30, 40, 50]):
        seed_torch(2024)
        data_path = f'.\\Dataset\\{dataset_name}\\fold5_data\\sub{sub_id}.npz'
        all_data = np.load(data_path)
        input_data, input_label = all_data['train_data'], all_data['train_label']

        input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(dim=1).to(device)
        print(input_data.shape)
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

        target_saliency = target_gradients[:, 0, 46, :].squeeze().cpu().numpy()
        target_saliency_mean = np.mean(target_saliency, axis=0)
        non_target_saliency = non_target_gradients[:, 0, 46, :].squeeze().cpu().numpy()
        non_target_saliency_mean = np.mean(non_target_saliency, axis=0)

        erp_data = input_data[target_index, 0, 46, :].detach().cpu().numpy().mean(axis=0)
        non_target_erp_data = input_data[non_target_index, 0, 46, :].detach().cpu().numpy().mean(axis=0)

        # Normalize the gradient
        target_saliency_data = target_saliency_mean
        target_saliency_data = ((target_saliency_data - target_saliency_data.min()) /
                                (target_saliency_data.max() - target_saliency_data.min())) * 2 - 1

        # Get the corresponding axis for the subplot
        ax = axes[i // 2, i % 2]

        # plot the saliency map
        # plt.imshow(target_saliency_mean, cmap='hot', interpolation='nearest')
        x = np.linspace(0, 1000, 128)
        ax.plot(x, target_saliency_data, color='black', label='Target Gradient', linestyle='--', linewidth=1)
        ax.plot(x, erp_data, color='red', label='Target', linewidth=1)
        ax.plot(x, non_target_erp_data, color='blue', label='Non-target', linewidth=1)
        ax.set_title(f"Subject {sub_id}")
        ax.set_xlabel("Time/ms")
        ax.set_ylabel("Magnitude")
        print('Finish!')

    # Adjust layout so that plots don't overlap
    fig.legend(['Target Gradient', 'Target', 'Non-target'], loc='lower center', ncol=3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('temporal_saliency_map.png', dpi=600, bbox_inches='tight')
    fig.savefig("temporal_saliency_map.pdf",
                dpi=600, bbox_inches='tight', format='pdf')
    plt.show()
