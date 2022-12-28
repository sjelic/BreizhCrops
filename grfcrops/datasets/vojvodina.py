
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from  breizhcrops.datasets.breizhcrops import get_default_target_transform


__all__ = ['VojvodinaDataset']

VOJVODINA_LABELS = {
            'barley' : 0,
            'maize' : 1,
            'rapeseed' : 2,
            'soya' : 3,
            'sugarbeet ' : 4,
            'sunflower' : 5,
            'wheat' : 6
}

def get_transform_label_to_code():
    return lambda y : VOJVODINA_LABELS[y]

def get_default_simple_transform():
    return lambda x: torch.from_numpy(x).type(torch.FloatTensor)



class VojvodinaDataset(Dataset):
    def __init__(self, country, name, data_dir='../data', transform = None, target_transform = None ):
        
        self.data_dir = data_dir
        self.country = country
        self.name = name
        self.y = np.load(os.path.join(data_dir,country,name, 'y.npy'))
    
        if transform is None:
            self.transform = lambda x : get_default_simple_transform()(x)
        else:
            self.transform = transform
        if target_transform is None:
            self.target_transform = lambda y : get_default_target_transform()(get_transform_label_to_code()(y))
        else:
            self.target_transform = target_transform
    
    def __len__(self):
        return len([ f for f in os.listdir(os.path.join(self.data_dir, self.country, self.name)) if f.endswith(".npy") ]) - 1
    
    def __getitem__(self, idx):
        x = np.load(os.path.join(self.data_dir,self.country,self.name, f'{idx}.npy'))
        return x, self.y, idx
    

