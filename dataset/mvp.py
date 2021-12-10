import os
import random
import sys
sys.path.append('.')

import h5py
import numpy as np
import torch

from torch.utils.data import Dataset

from visualization import plot_pcd_one_view


class MVP(Dataset):
    def __init__(self, data_root, category, split, npoints):
        super().__init__()

        self.cat2labels = {'plane':   0,
                           'dresser': 1,
                           'car':     2,
                           'chair':   3,
                           'lamp':    4,
                           'sofa':    5,
                           'table':   6,
                           'boat':    7}

        assert split in ['train', 'test'], "illegal spliting, only can be 'train' or 'test'"
        assert npoints in [2048, 4096, 8192, 16384], "illegal npoints, only can be 2048, 4096, 8192 or 16384"

        self.data_root = data_root
        self.category = category
        self.split = split
        self.npoints = npoints

        self.partials, self.completes = self._load_data()
    
    def __getitem__(self, index):
        partial = self.partials[index]
        complete = self.completes[index // 26]
        return torch.from_numpy(partial), torch.from_numpy(complete)

    def __len__(self):
        return len(self.partials)

    def _load_data(self):
        partial_file = os.path.join(self.data_root, 'mvp_{}_input.h5'.format(self.split))
        complete_file = os.path.join(self.data_root, 'mvp_{}_gt_{}pts.h5'.format(self.split, self.npoints))

        if self.category != 'all':
            with h5py.File(partial_file, 'r') as fp:
                partial_pcs =  np.array(fp['incomplete_pcds'])[np.array(fp['labels']) == self.cat2labels[self.category]]
            with h5py.File(complete_file, 'r') as fc:
                complete_pcs =  np.array(fc['complete_pcds'])[np.array(fc['labels']) == self.cat2labels[self.category]]
        else:
            with h5py.File(partial_file, 'r') as fp:
                partial_pcs =  np.array(fp['incomplete_pcds'])
            with h5py.File(complete_file, 'r') as fc:
                complete_pcs =  np.array(fc['complete_pcds'])
        
        return partial_pcs, complete_pcs


class MVP_v2(Dataset):
    def __init__(self, data_root, category, split, npoints):
        super().__init__()

        self.cat2labels = {'plane':   0,
                           'dresser': 1,
                           'car':     2,
                           'chair':   3,
                           'lamp':    4,
                           'sofa':    5,
                           'table':   6,
                           'boat':    7}

        assert split in ['train', 'test'], "illegal spliting, only can be 'train' or 'test'"
        assert npoints in [2048, 4096, 8192, 16384], "illegal npoints, only can be 2048, 4096, 8192 or 16384"

        self.data_root = data_root
        self.category = category
        self.split = split
        self.npoints = npoints

        self.partials, self.completes = self._load_data()
    
    def __getitem__(self, index):
        partial = self.partials[random.randint(index * 26, (index + 1) * 26 - 1)]
        complete = self.completes[index]
        return torch.from_numpy(partial), torch.from_numpy(complete)

    def __len__(self):
        return len(self.completes)

    def _load_data(self):
        partial_file = os.path.join(self.data_root, 'mvp_{}_input.h5'.format(self.split))
        complete_file = os.path.join(self.data_root, 'mvp_{}_gt_{}pts.h5'.format(self.split, self.npoints))

        if self.category != 'all':
            with h5py.File(partial_file, 'r') as fp:
                partial_pcs =  np.array(fp['incomplete_pcds'])[np.array(fp['labels']) == self.cat2labels[self.category]]
            with h5py.File(complete_file, 'r') as fc:
                complete_pcs =  np.array(fc['complete_pcds'])[np.array(fc['labels']) == self.cat2labels[self.category]]
        else:
            with h5py.File(partial_file, 'r') as fp:
                partial_pcs =  np.array(fp['incomplete_pcds'])
            with h5py.File(complete_file, 'r') as fc:
                complete_pcs =  np.array(fc['complete_pcds'])
        
        return partial_pcs, complete_pcs


if __name__ == '__main__':
    dataset = MVP('/root/autodl-tmp/data/MVP', category='chair', split='train', npoints=8192)
    incomplete, complete = dataset[random.randint(0, len(dataset) - 1)]
    incomplete = incomplete.numpy()
    complete = complete.numpy()
    plot_pcd_one_view('temp/temp1.png', [incomplete, complete], ['incomplete', 'complete'])

    dataset = MVP_v2('/root/autodl-tmp/data/MVP', category='chair', split='train', npoints=8192)
    incomplete, complete = dataset[random.randint(0, len(dataset) - 1)]
    incomplete = incomplete.numpy()
    complete = complete.numpy()
    plot_pcd_one_view('temp/temp2.png', [incomplete, complete], ['incomplete', 'complete'])
