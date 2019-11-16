from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import imageio


class MiniDataset(Dataset):

    def __init__(self, root_dir, trans=None, num_to_holdout = 50):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.size_of_dataset = len(os.listdir(root_dir)) - num_to_holdout
        self.num_to_holdout = num_to_holdout
        if self.size_of_dataset < 1:
            print("dataset is not larger than size of holdout")
        self.root_dir = root_dir
        #self.file_names = [f for f in os.listdir(root_dir) if not f.startswith('.')]
        self.transform = trans
        self.last_idx = -1
        self.last_heldout_idx = -1
        self.random_permutation = np.random.permutation(np.arange(self.size_of_dataset))
        self.random_permutation_heldout = np.random.permutation(np.arange(self.num_to_holdout))

    def __len__(self):

        return self.size_of_dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                str(idx) + ".jpg")
        sample = imageio.imread(img_name)

        sample = sample / 255.0

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __getrandomitem__(self):
        # return random sample
        if self.last_idx + 1 == self.size_of_dataset:
            #print("Reset data set and shuffle")
            self.reset()

        self.last_idx += 1

        idx = self.random_permutation[self.last_idx]

        return self.__getitem__(idx)

    def __getrandomheldoutitem__(self):

        if self.last_heldout_idx + 1 == self.num_to_holdout:
            #print("Reset data set and shuffle")
            self.reset_heldout()

        self.last_heldout_idx += 1

        idx = self.random_permutation_heldout[self.last_heldout_idx]

        img_name = os.path.join(self.root_dir,
                                str(self.size_of_dataset + idx) + ".jpg")

        sample = imageio.imread(img_name)

        sample = sample / 255.0

        if self.transform:
            sample = self.transform(sample)

        return sample

    def reset(self):
        self.last_idx = -1
        self.random_permutation = np.random.permutation(np.arange(self.size_of_dataset))
    def reset_heldout(self):
        self.last_heldout_idx = -1
        self.random_permutation_heldout = np.random.permutation(np.arange(self.num_to_holdout))



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).cuda()

