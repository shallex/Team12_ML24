import torch
import torchvision
import matplotlib.pyplot as plt

class MnistDataloader(torch.utils.data.DataLoader):
    def __init__(self, batch_size=128, shuffle=True, num_workers=0, train=True, download=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataset = torchvision.datasets.MNIST('./files/', train=train, download=download,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
        
        super(MnistDataloader, self).__init__(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def show_data(self):
        fig = plt.figure()
        for i in range(6):
            img = self.dataset[i][0].squeeze(0)
            target = self.dataset[i][1]
            plt.subplot(2,3,i+1)
            plt.tight_layout()
            plt.imshow(img, cmap='gray', interpolation='none')
            plt.title("Ground Truth: {}".format(target))
            plt.xticks([])
            plt.yticks([])