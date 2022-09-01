import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
import torchvision

class DataRawPatches(Dataset):
    def __init__(self, data_path, ps=16, transform=None, target_transform=None):
        X = np.array(io.imread(data_path)).astype(float)
        for i in range(X.shape[0]):
            f = X[i,:,:]
            i99 = np.percentile(f, 99)
            f[f > i99] = i99
            max_f = f.max()
            min_f = f.min()
            X[i,:,:] = (f - min_f) / (max_f - min_f)

        self.data = X

        if transform: 
            self.transform = transform
        else:
            # self.transform = torchvision.transforms.Compose([torchvision.transforms.RandomVerticalFlip(p=0.5),torchvision.transforms.RandomHorizontalFlip(p=0.5),torchvision.transforms.RandomCrop(size=ps)])
            self.transform = torchvision.transforms.Compose([torchvision.transforms.RandomVerticalFlip(p=0.5),torchvision.transforms.RandomHorizontalFlip(p=0.5)])
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image = torch.tensor(self.data[idx,:,:]).unsqueeze_(0)
        if self.transform:
            image = self.transform(image)
        # max_f = image.max()
        # min_f = image.min()
        # image = (image- min_f) / (max_f - min_f)
        label = image
        if self.target_transform:
            label = self.target_transform(image)
        return image, label