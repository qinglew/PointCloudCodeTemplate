import os
import random
import h5py as h5

import torch
import torch.utils.data as data
import numpy as np


class Completion3D(data.Dataset):
    def __init__(self, data_root, category, split):
        super().__init__()
        self.data_root = data_root
        self.category = category
        self.split = split

        self.cat2id = {'boat'   : '04530566',
                       'car'    : '02958343',
                       'chair'  : '03001627',
                       'dresser': '02933112',
                       'lamp'   : '03636649',
                       'plane'  : '02691156',
                       'sofa'   : '04256520',
                       'table'  : '04379243'}

        self.partial_pcs, self.complete_pcs = self._load_data()
    
    def __getitem__(self, index):
        return torch.from_numpy(self.partial_pcs[index]), torch.from_numpy(self.complete_pcs[index])

    def __len__(self):
        return len(self.partial_pcs)

    def _load_data(self):
        complete_dir = os.path.join(self.data_root, self.split, 'gt', self.cat2id[self.category])

        partial_pcs = list()
        complete_pcs = list()

        for filename in os.listdir(complete_dir):
            complete_path = os.path.join(complete_dir, filename)
            complete_pcs.append(self._load_h5(complete_path))
            partial_pcs.append(self._load_h5(complete_path.replace('gt', 'partial')))
        
        return partial_pcs, complete_pcs
    
    def _load_h5(self, filename):
        with h5.File(filename, 'r') as f:
            pc_array = np.asarray(f['data'], dtype=np.float32)
        return pc_array


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from visualization.visualization import plot_pcd_one_view
    dataset = Completion3D('/root/autodl-tmp/data/Completion3D', 'chair', 'val')
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    for epoch in range(2):
        for partials, completes in dataloader:
            plot_pcd_one_view('../temp/{}.png'.format(epoch), [partials[0].numpy(), completes[0].numpy()], ['partial', 'complete'])
            break
