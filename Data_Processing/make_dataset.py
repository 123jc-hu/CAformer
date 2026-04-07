import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def dataloader(x_data, y_data, batch_size=128, shuffle=True, num_workers=0, pin_memory=False, transform=None):
    torch.manual_seed(2024)
    x_data = torch.as_tensor(np.array(x_data), dtype=torch.float32)
    y_data = torch.as_tensor(np.array(y_data), dtype=torch.int64)
    if len(x_data.shape) == 3:
        x_data = x_data.unsqueeze(dim=1)
    dataset = TensorDataset(x_data, y_data)
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
