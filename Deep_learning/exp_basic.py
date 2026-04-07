import torch

from Models.ablation_study import CAFormer


class ExpBasic:
    def __init__(self, args):
        self.args = args
        self.model_registry = {
            "CAFormer": CAFormer,
            "TTMTN": CAFormer,
        }
        self.model_dict = self.model_registry
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device("cuda:0")
            print("Use GPU: cuda:0")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device


BaseExperiment = ExpBasic
