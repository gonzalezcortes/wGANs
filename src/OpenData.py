
import torch
import torchvision
#from torchvision.datasets import MNIST 

class FilterDf(torchvision.datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super(FilterDf, self).__init__(*args, **kwargs)

        idx = (self.targets == 8)  # Get the index of the digit '8' samples
        self.data = self.data[idx]
        self.targets = self.targets[idx]

class DataLoaderCreator:
    def __init__(self, root='./data', batch_size=32, download=True, shuffle={'train': True, 'test': False}):
        self.root = root
        self.batch_size = batch_size
        self.download = download
        self.shuffle = shuffle
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
        
    def get_loader(self, train):
        dataset = FilterDf(root=self.root, train=train, download=self.download, transform=self.transform)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle['train' if train else 'test'])

    @property
    def trainloader(self):
        return self.get_loader(train=True)
    
    @property
    def testloader(self):
        return self.get_loader(train=False)
