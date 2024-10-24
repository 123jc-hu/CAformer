import os
import torch
from Models import EEGNet, DeepConvNet, PLNet, EEGInception, MGIFNet, CNN_LSTM, P3Net, EEGNet_regions, PPNN, MTCN
from Models.Transformer import (CNN_Transformer, iTransformer, PatchTST, EEG_Conformer, CNN_Transformer_regions, CTMTN,
                                CTMTN_handle)
from Models.ablation_study import TTMTN, TTMTN_T, TTMTN_C, TTMTN_T_C


class ExpBasic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "EEGNet": EEGNet,
            "DeepConvNet": DeepConvNet,
            "PLNet": PLNet,
            "EEGInception": EEGInception,
            "MGIFNet": MGIFNet,
            "CNN_LSTM": CNN_LSTM,
            "P3Net": P3Net,
            "EEGNet_regions": EEGNet_regions,
            "CNN_Transformer": CNN_Transformer,
            "iTransformer": iTransformer,
            "PatchTST": PatchTST,
            "EEG_Conformer": EEG_Conformer,
            "CNN_Transformer_regions": CNN_Transformer_regions,
            "PPNN": PPNN,
            "CTMTN": CTMTN,
            "MTCN": MTCN,
            "CTMTN_handle": CTMTN_handle,
            "TTMTN": TTMTN,
            "TTMTN_T": TTMTN_T,
            "TTMTN_C": TTMTN_C,
            "TTMTN_T_C": TTMTN_T_C,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:0')
            print('Use GPU: cuda:0')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    # def vali(self):
    #     pass

    # def train(self):
    #     pass
    #
    # def test(self):
    #     pass
