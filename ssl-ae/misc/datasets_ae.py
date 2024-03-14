import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision

class AutoencoderDataset(torch.utils.data.Dataset):
    """
    Construct dataset for autoencoder training from another dataset.
    """
    
    def __init__(self, dataset, dim: int=0):
        """
        Initialization.
        
        Parameters
        ----------
        dataset
            The dataset from which to make dataset for the autoencoder.
        dim : int, optional
            The number of the subelement (in each entry) to be repeated.
        """
        
        self.dataset = dataset
        self.dim = dim
        
        
    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, index):
        x = self.dataset[index][self.dim]
        return (x, x)
        
        
class MNIST_w_imagesums(Dataset):
    def __init__(self, dataset, aug_ratio=0.1) -> None:
        self.dataset = dataset
        self.dd_idxs_aug = {}
        
        self.idxs_aug = (np.arange(len(self.dataset)))
        np.random.shuffle(self.idxs_aug)
        # copying for saving indices
        self.idxs_aug_full = self.idxs_aug.copy()
        
        len_aug = int(aug_ratio*len(self.idxs_aug))
        # check even length of len_aug
        if len_aug % 2 == 0:
            pass
        else:
            len_aug -= 1
            
        idxs_aug_only = self.idxs_aug[-len_aug:] # choose last indices for augs
        for l,r in zip(idxs_aug_only[:len_aug//2], idxs_aug_only[len_aug//2:]):
            self.dd_idxs_aug[l] = r
            self.dd_idxs_aug[r] = l
            
        self.idxs_aug = self.idxs_aug[:-len_aug//2]
        
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.idxs_aug[index] in self.dd_idxs_aug:
            img_1, target_1 = self.dataset[self.idxs_aug[index]]
            img_2, target_2 = self.dataset[self.dd_idxs_aug[self.idxs_aug[index]]]
            
            img = img_1 + img_2
            target = str(target_1)+'+'+str(target_2)
            
            # (0,), (1.58,) was chosen base on computed statistics - see 
            # see commented celll in "ae-reconstruction-L1-imagesums-lossSamples.ipynb" for code
            img=torchvision.transforms.Normalize((0,), (1.58,))(img)
        else:
            img, target = self.dataset[self.idxs_aug[index]]
        
        return img, target
    

    def __len__(self) -> int:
        return len(self.idxs_aug)