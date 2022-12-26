
from torch.utils.data import Dataset
import torch
import numpy as np
from  breizhcrops.datasets.breizhcrops import get_default_target_transform


__all__ = ['VojvodinaDataset']

VOJVODINA_LABELS = {
            'barley' : 0,
            'maize' : 1,
            'not_defined' : 2,
            'olives' : 3,
            'rapeseed' : 4,
            'soya' : 5,
            'sugarbeet' : 6,
            'sunflower' : 7,
            'wheat' : 8
}

def get_transform_label_to_code():
    return lambda y : VOJVODINA_LABELS[y]

def get_default_simple_transform():
    return lambda x: torch.from_numpy(x).type(torch.FloatTensor)



class VojvodinaDataset(Dataset):
    def __init__(self, X_path = '../data/X_sr.npy', y_path = '../data/y_sr.npy', transform = None, target_transform = None ):
        
        
        
        self.X = np.load(X_path)
        self.y = np.load(y_path)
    
        if transform is None:
            self.transform = lambda x : get_default_simple_transform()(x)
        else:
            self.transform = transform
        if target_transform is None:
            self.target_transform = lambda y : get_default_target_transform()(get_transform_label_to_code()(y))
        else:
            self.target_transform = target_transform
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.transform(self.X[idx])
        y = self.target_transform(self.y[idx])
        return x, y, idx
    

