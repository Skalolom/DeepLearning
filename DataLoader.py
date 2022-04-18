import pandas as pd
import numpy as np
from torch import Tensor
import torch
import os
import warnings

warnings.filterwarnings("ignore")  # avoid printing out absolute paths
from datetime import date, timedelta
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):

    def __init__(self, df: pd.DataFrame, encoder_len: int, decoder_len: int, targets: list, reals=None,
                 categoricals=None, time_var=None):
        self.df = df
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.targets = targets
        self.reals = reals
        self.categoricals = categoricals
        self.time_var = time_var

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        """
        Get data window of  size = decoder + encoder length from the given element index
        :param idx:
        :return:
        """

        def get_window(idx_start: int, idx_end: int):

            """
            Get data window of  size = decoder + encoder length
            :param self:
            :param idx_start: index of the first element in the window
            :param idx_end: index of the last element in the window
            :return:
            """

            tensor_arr = []

            targets_tensor = torch.Tensor(size=(self.encoder_len + self.decoder_len, len(self.targets)))
            targets_tensor[:, :] = torch.from_numpy(self.df.iloc[idx_start: idx_end][self.targets].values).to(targets_tensor)
            tensor_arr.append(targets_tensor)

            if self.reals:
                conts_tensor = torch.Tensor(size=(self.encoder_len + self.decoder_len, len(self.reals)))
                conts_tensor[:, :] = torch.from_numpy(self.df.iloc[idx_start: idx_end][self.reals].values).to(conts_tensor)
                tensor_arr.append(conts_tensor)

            if self.categoricals:
                cats_tensor = torch.Tensor(size=(self.encoder_len + self.decoder_len, len(self.categoricals)))
                cats_tensor[:, :] = torch.from_numpy(self.df.iloc[idx_start: idx_end][self.categoricals].values).to(cats_tensor)
                tensor_arr.append(cats_tensor)

            window = torch.cat(tensor_arr, 1)

            return window

        last_data_element_idx = self.df.shape[0] - 1
        window_size = self.encoder_len + self.decoder_len
        begin_idx = idx - self.encoder_len
        end_idx = idx + self.decoder_len

        if begin_idx < 0:
            # return window starting from idx=0
            sample = get_window(0, window_size)
        elif end_idx > last_data_element_idx:
            # return window, ending on the last element in the dataset
            sample = get_window(last_data_element_idx - window_size, last_data_element_idx)
        else:
            sample = get_window(begin_idx, end_idx)

        return sample