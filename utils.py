import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib
import numpy as np
from pathlib import Path
from multiprocessing import Pool


def load_MRI_file(path: Path | str, cross_section: bool = False):
    img = nib.load(path)

    raw = img.get_fdata()
    raw = raw / raw.mean()
    raw = raw.reshape((1, *raw.shape))

    if cross_section:
        return (
            raw[:, raw.shape[1] // 2, :, :],
            raw[:, :, raw.shape[2] // 2, :],
            raw[:, :, :, raw.shape[3] // 2],
        )
    else:
        return raw


class MRIDataset(Dataset):
    workers = 8

    def __init__(self, df: pd.DataFrame):
        scores = df["score"].to_numpy()
        self.length = len(scores)

        self.x = []
        for i in range(self.length):
            filename = df["filename"].iloc[i]
            self.x.append(load_MRI_file(filename))

        self.y = np.zeros((self.length, 2))
        for i in range(self.length):
            j = 1 if scores[i] == 3 else 0
            self.y[i][j] = 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx])
        y = torch.FloatTensor(self.y[idx])

        return x, y


class MRICrossSectionDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        scores = df["score"].to_numpy()
        self.length = len(scores)

        self.xd = []
        self.xh = []
        self.xw = []
        for i in range(self.length):
            filename = df["filename"].iloc[i]
            xd, xh, xw = load_MRI_file(filename, cross_section=True)
            self.xd.append(xd)
            self.xh.append(xh)
            self.xw.append(xw)

        self.y = np.zeros((self.length, 2))
        for i in range(self.length):
            j = 1 if scores[i] == 3 else 0
            self.y[i][j] = 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        xd = torch.FloatTensor(self.xd[idx])
        xh = torch.FloatTensor(self.xh[idx])
        xw = torch.FloatTensor(self.xw[idx])

        y = torch.FloatTensor(self.y[idx])

        return xd, xh, xw, y


def init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(
        m,
        (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        ),
    ):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class EarlyStopping:
    def __init__(
        self,
        patience: int = 2,
        delta: float = 0,
        path: str | Path = "ephemeral_best_weights.pth",
    ):
        self.min_loss = np.inf
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0

    def evaluate(self, model: nn.Module, loss: float):
        if loss + self.delta > self.min_loss:
            self.counter += 1
        else:
            self.counter = 0

        if loss < self.min_loss:
            self.min_loss = loss
            torch.save(model.state_dict(), self.path)

    def should_stop(self):
        return self.counter >= self.patience

    def load_best(self, model: nn.Module):
        model.load_state_dict(torch.load(self.path))
        return model
