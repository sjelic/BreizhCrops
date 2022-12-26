import numpy as np
from datasets.vojvodina import VojvodinaDataset
from breizhcrops.models.TransformerModel import TransformerModel
import torch

# X_sr = np.load('data/X_sr.npy')
# X_sr = torch.from_numpy(X_sr).to(torch.float32)
# y_sr = np.load('data/y_sr.npy')

dataset = VojvodinaDataset()


transformer = TransformerModel(input_dim=48, num_classes=9, d_model=40, n_head=2, n_layers=5,
                 d_inner=128, activation="relu", dropout=0.017998950510888446)
x, y = dataset.__getitem__(0)
x_trans = transformer.forward(x)