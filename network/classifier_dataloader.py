import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
import string
from einops import rearrange, repeat
import pickle


class Dataset(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """
    def __init__(self, path_layer1, path_layer2):
        super(Dataset, self).__init__()
        with open(path_layer1, 'rb') as f:
            data1 = pickle.load(f)
        self.data1 = data
        with open(path_layer2, 'rb') as f:
            data2 = pickle.load(f)
        self.data1 = data1
        self.data2 = data2

    def __getitem__(self, index):
        if self.data1[index][1] != self.data2[index][1]:
            assert 'label error'

        return self.data1[index][0], self.data2[index][0], self.data1[index][1]

    def __len__(self):
        return len(self.data1)
