import torch
import numpy as np
import open3d as o3d

from metrics.loss import CD, EMD


class Metric():
    def __init__(self, names, num):
        self.metrics = {k: 0. for k in names}
        self.num = num
    
    def reset(self):
        self.metrics = {k: 0. for k in self.metrics}
    
    def update(self, d):
        for k in d:
            self.metrics[k] += d[k]
    
    def mean(self):
        for k in self.metrics:
            self.metrics[k] = self.metrics[k] / self.num
        return self.metrics
        

def directed_cd(pcs1, pcs2):
    dist1, _ = CD(pcs1, pcs2)
    dist1 = torch.sum(torch.mean(dist1, dim=1))
    return dist1


def l2_cd(pcs1, pcs2):
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.mean(dist1, dim=1)
    dist2 = torch.mean(dist2, dim=1)
    return torch.sum(dist1 + dist2)


def l1_cd(pcs1, pcs2):
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.mean(torch.sqrt(dist1), 1)
    dist2 = torch.mean(torch.sqrt(dist2), 1)
    return torch.sum(dist1 + dist2) / 2


def emd(pcs1, pcs2):
    dists = EMD(pcs1, pcs2)
    return torch.sum(dists)


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Accuracy, Completioness and F1-Score in PCL2PCL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def accuracy(P_recon, P_gt, thre=0.01):
    """
    ACCURACY
    P_gt: N x 3, np array
    P_recon: N x 3, np array
    """
    npoint = P_recon.shape[0]

    P_recon_here = np.expand_dims(P_recon, axis=1) # N x 1 x 3
    P_recon_here = np.tile(P_recon_here, (1, npoint, 1)) # N x N x 3

    P_gt_here = np.tile(P_gt, (npoint,1)) 
    P_gt_here =  np.reshape(P_gt_here, (npoint, npoint, 3)) # N x N x 3

    dists = np.linalg.norm(P_recon_here - P_gt_here, axis=-1) # N x N x 1
    dists = np.squeeze(dists) # N x N

    min_dists = np.amin(dists, axis=1) # 1 x N

    avg_dist = np.mean(min_dists)

    matched = min_dists[min_dists < thre]
    fraction = matched.shape[0] / npoint
    return fraction, avg_dist


def accuracy_cuda(P_recon, P_gt, thre=0.01):
    """
    cuda version of accuracy
    """
    npoint = P_recon.shape[0]
    if isinstance(P_gt, np.ndarray):
        P_recon = torch.from_numpy(P_recon).cuda().unsqueeze(0)
        P_gt = torch.from_numpy(P_gt).cuda().unsqueeze(0)
    else:
        P_recon = P_recon.unsqueeze(0)
        P_gt = P_gt.unsqueeze(0)
    P_recon_here = P_recon.unsqueeze(2).repeat(1,1,npoint,1)
    P_gt_here = P_gt.unsqueeze(1).repeat(1,npoint,1,1)
    
    dist = P_recon_here.add(-P_gt_here)
    dist_value = torch.norm(dist,dim=3).squeeze(0)

    min_dists, _ = dist_value.min(axis=1)
    avg_dist = min_dists.mean()
    
    matched = min_dists[min_dists < thre]
    fraction = matched.shape[0] / npoint
    return fraction, avg_dist


def completeness(P_recon, P_gt, thre=0.01):
    '''
    COMPLETENESS
    P_gt: N x 3, np array
    P_recon: N x 3, np array
    '''

    npoint = P_recon.shape[0]

    P_gt_here = np.expand_dims(P_gt, axis=1) # N x 1 x 3
    P_gt_here = np.tile(P_gt_here, (1, npoint, 1)) # N x N x 3

    P_recon_here = np.tile(P_recon, (npoint,1)) 
    P_recon_here =  np.reshape(P_recon_here, (npoint, npoint, 3)) # N x N x 3

    dists = np.linalg.norm(P_gt_here - P_recon_here, axis=-1) # N x N x 1
    dists = np.squeeze(dists) # N x N

    min_dists = np.amin(dists, axis=1) # N x 1
    
    avg_min_dist = np.mean(min_dists)

    matched = min_dists[min_dists < thre]
    fraction = matched.shape[0] / npoint
    return fraction, avg_min_dist


def completeness_cuda(P_recon, P_gt, thre=0.01):
    """
    completeness_cuda
    """
    npoint = P_recon.shape[0]
    if isinstance(P_gt, np.ndarray):
        P_recon = torch.from_numpy(P_recon).cuda().unsqueeze(0)
        P_gt = torch.from_numpy(P_gt).cuda().unsqueeze(0)
    else:
        P_recon = P_recon.unsqueeze(0)
        P_gt = P_gt.unsqueeze(0)
    P_recon_here = P_recon.unsqueeze(2).repeat(1,1,npoint,1)
    P_gt_here = P_gt.unsqueeze(1).repeat(1,npoint,1,1)
    dist = P_gt_here.add(-P_recon_here)
    dist_value = torch.norm(dist,dim=3).squeeze(0)

    min_dists, _ = dist_value.min(axis=0)
    avg_dist = min_dists.mean()
    
    matched = min_dists[min_dists < thre]
    fraction = matched.shape[0] / npoint
    return fraction, avg_dist


def F1_score(precision, recall):
    f = 2 * precision * recall / (precision + recall)
    return f

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Accuracy, Completioness and F1-Score in PCL2PCL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def f_score(pred, gt, th=0.01):
    """
    References: https://github.com/lmb-freiburg/what3d/blob/master/util.py

    Args:
        pred (np.ndarray): (N1, 3)
        gt   (np.ndarray): (N2, 3)
        th   (float): a distance threshhold
    """
    pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred))
    gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))

    dist1 = pred.compute_point_cloud_distance(gt)
    dist2 = gt.compute_point_cloud_distance(pred)

    recall = float(sum(d < th for d in dist2)) / float(len(dist2))
    precision = float(sum(d < th for d in dist1)) / float(len(dist1))
    return 2 * recall * precision / (recall + precision) if recall + precision else 0
