from math import ceil
import torch
from torch.utils.data import Dataset
from skimage import io
from sklearn.feature_extraction import image
import numpy as np
import torch.nn.functional as F

class DataPatchesGT(Dataset):
    def __init__(self, data_path, gt_path, ps=64, transform=None, target_transform=None):
        X = np.array(io.imread(data_path)).astype(float)
        Y = np.array(io.imread(gt_path)).astype(float)
        for i in range(X.shape[0]):
            f = X[i,:,:]
            max_f = f.max()
            min_f = f.min()
            X[i,:,:] = (f - min_f) / (max_f - min_f)

        # self.inputs = X
        # self.targets = Y
        scale_factor = Y.shape[1]/X.shape[1]
        X_interp = F.interpolate(torch.tensor(X).unsqueeze(1), scale_factor=scale_factor, mode='bicubic').squeeze_(1).cpu().detach().numpy()
        X_interp = np.expand_dims(X_interp, 1)
        Y = np.expand_dims(Y, 1)
        Z = np.concatenate([X_interp,Y], axis=1)
        patches_arr = []
        for i in range(Z.shape[0]): 
            patches = image.extract_patches_2d(Z[i,:,:,:].transpose((2,1,0)), patch_size=(ps, ps), max_patches=40)
            patches_arr.append(patches.transpose((0,3,2,1)))   
        data = np.concatenate(patches_arr, axis=0)
        self.data = data

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # image = torch.tensor(self.inputs[idx,:,:]).unsqueeze_(0)
        # label = torch.tensor(self.targets[idx,:,:]).unsqueeze_(0)
        image = torch.tensor(self.data[idx,0,:,:]).unsqueeze_(0)
        label = torch.tensor(self.data[idx,1,:,:]).unsqueeze_(0)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(image)
        return image, label