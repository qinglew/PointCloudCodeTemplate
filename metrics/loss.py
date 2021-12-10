import torch
import torch.nn as nn

from torch.autograd import grad

from extensions.chamfer_distance.chamfer_distance import ChamferDistance
from extensions.earth_movers_distance.emd import EarthMoverDistance


CD = ChamferDistance()
EMD = EarthMoverDistance()


class GANLoss(nn.Module):
    """Module for computing the GAN loss for the generator.

    When `use_least_squares` is turned on, we use mean squared error loss,
    otherwise we use the standard binary cross-entropy loss.

    Note: We use the convention that the discriminator predicts the probability
    that the target image is real. Therefore real corresponds to label 1.0."""
    def __init__(self, device, use_least_squares=False):
        super(GANLoss, self).__init__()
        self.loss_fn = nn.MSELoss() if use_least_squares else nn.BCEWithLogitsLoss()
        self.real_label = None  # Label tensor passed to loss if target is real
        self.fake_label = None  # Label tensor passed to loss if target is fake
        self.device = device

    def _get_label_tensor(self, input_, is_tgt_real):
        # Create label tensor if needed
        if is_tgt_real and (self.real_label is None or self.real_label.numel() != input_.numel()):
            self.real_label = torch.ones_like(input_, device=self.device, requires_grad=False)
        elif not is_tgt_real and (self.fake_label is None or self.fake_label.numel() != input_.numel()):
            self.fake_label = torch.zeros_like(input_, device=self.device, requires_grad=False)

        return self.real_label if is_tgt_real else self.fake_label

    def __call__(self, input_, is_tgt_real):
        label = self._get_label_tensor(input_, is_tgt_real)
        return self.loss_fn(input_, label)

    def forward(self, input_):
        raise NotImplementedError('GANLoss should be called directly.')


def directed_cd_loss(pcs1, pcs2):
    """
    Directed Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, _ = CD(pcs1, pcs2)
    return torch.mean(dist1)


def cd_loss_L1(pcs1, pcs2):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    return (torch.mean(dist1) + torch.mean(dist2)) / 2.0


def cd_loss_L2(pcs1, pcs2):
    """
    L2 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    return torch.mean(dist1) + torch.mean(dist2)


def emd_loss(pcs1, pcs2):
    """
    EMD Loss.

    Args:
        xyz1 (torch.Tensor): (b, N, 3)
        xyz2 (torch.Tensor): (b, N, 3)
    """
    dists = EMD(pcs1, pcs2)
    return torch.mean(dists)


def gradient_penalty(net_D, real_data, fake_data):
    batch_size = real_data.size(0)
    device = real_data.device
    alpha = torch.rand(real_data.size(0), 1, 1, requires_grad=True).to(device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    prob_interpolated = net_D(interpolated)
    gradients = grad(outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                     create_graph=True, retain_graph=True, only_inputs=True)[0].contiguous().view(batch_size,-1)
    return gradients.norm(2, dim=1).mean()
