from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

class MNISTDataDictWrapper(Dataset):
    def __init__(self, dset, phase):
        super().__init__()
        self.dset = dset
        self.phase = phase
    def __getitem__(self, i):
        x, y = self.dset[i] # x shape: 1,28,28, y: digital label
        return {"jpg": x, "cls": y}

    def __len__(self):
        if self.phase == 'val':
            return 1000
        return len(self.dset)

class MNIST_dataset():
    def __init__(self, phase):

        self.train_dset = MNISTDataDictWrapper(
            torchvision.datasets.MNIST(
                root=".data/",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)]
                ),
            ), phase
        )
       
        self.val_dset = MNISTDataDictWrapper(
            torchvision.datasets.MNIST(
                root=".data/",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)]
                ),
                ), phase
            )
        
        self.test_dset = MNISTDataDictWrapper(
            torchvision.datasets.MNIST(
                root=".data/",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)]
                ),
            ), phase
        )
        