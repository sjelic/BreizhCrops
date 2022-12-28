from datasets.vojvodina import VojvodinaDataset
from breizhcrops.datasets.breizhcrops import get_default_target_transform

__all__ = ['BretagneDataset']

class BretagneDataset(VojvodinaDataset):
    def __init__(self, country, name, data_dir='../data', transform = None, target_transform = None ):
        super().__init__(country=country, name=name, data_dir=data_dir, transform=transform, target_transform=target_transform)
        if target_transform is None:
            self.target_transform = lambda y : get_default_target_transform()(y)
        else:
            self.target_transform = target_transform
            

