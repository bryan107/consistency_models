from torch.utils.data import Dataset
from sklearn.datasets import make_swiss_roll
from torch.distributions import Normal, Categorical, MixtureSameFamily
import torch

class _1D_Dataset(Dataset):
    def __init__(self, data_size = 10000, phase='test'):
        super(_1D_Dataset, self).__init__()
        mix = Categorical(torch.ones(2,))
        comp = Normal(torch.tensor([0 - 2., 0 + 2.]), torch.tensor([.5, .5]))
        data_dist = MixtureSameFamily(mix, comp)

        data = data_dist.sample([data_size])
        if phase == 'train':
            self.size = int(80.0 / 100.0 * len(data))
            self.data = data[:self.size].unsqueeze(1).numpy()
        elif phase == 'test':
            self.size = int(20.0 / 100.0 * len(data))
            self.data = data[-self.size:].unsqueeze(1).numpy()
        elif phase == 'val':
            self.size = int(1.0 / 100.0 * len(data))
            self.data = data[-self.size:].unsqueeze(1).numpy()

    
    
    def __len__(self):
        return self.size

    def __iter__(self):
        return self

    def __getitem__(self, index):
        
        return self.data[index]
    
    