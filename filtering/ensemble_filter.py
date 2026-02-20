import torch
from pytorch3d.ops import knn_points

from utils.graphics_utils import get_intrinsic


# This function is modified from an original verison from "https://github.com/sailor-z/SE-GS/blob/main/train_llff.py#L292"
def find_reset_gaussians_ratio(gaussians, view, radii, variance, ratio=0.1):
    pts3d = gaussians.get_xyz

    K = torch.from_numpy(get_intrinsic(view)).cuda().float()
    R = torch.from_numpy(view.R.transpose()).cuda().float()
    T = torch.from_numpy(view.T[:, None]).cuda().float()

    proj_pts3d = K @ (R @ pts3d.transpose(0, 1) + T)
    proj_pts3d = proj_pts3d.transpose(0, 1)
    depth = proj_pts3d[:, 2]
    proj_pts = proj_pts3d[:, :2] / depth[:, None].clamp(min=1e-6)

    valid_maskx = (proj_pts[:, 0] >= 0) & (proj_pts[:, 0] < view.image_width)
    valid_masky = (proj_pts[:, 1] >= 0) & (proj_pts[:, 1] < view.image_height)
    valid_mask = torch.logical_and(valid_maskx, valid_masky)

    variance = variance.mean(dim=0)   # shape becomes [H, W]
    variance_sorted = torch.sort(variance.flatten(0), descending=True)[0]
    print("Ratio: ", ratio)
    thr = variance_sorted[int(ratio * view.image_width * view.image_height)] #TODO Change to quantile


    std_mask = variance > thr

    if torch.all(std_mask == 0):
        return pts3d.new_zeros(pts3d.shape[0])

    pts2d = torch.stack(torch.meshgrid(torch.arange(view.image_height, device="cuda"), torch.arange(view.image_width, device="cuda")), -1).float()
    pts2d = pts2d[..., (1, 0)] 
    pts2d = pts2d[std_mask]

    dist, _, _ = knn_points(proj_pts[None], pts2d[None], K=1)
    dist = dist.squeeze()

    reset_mask = dist < radii
    reset_mask = torch.logical_and(valid_mask, reset_mask)

    return reset_mask

# This function is modified from an original verison from "https://github.com/sailor-z/SE-GS/blob/main/train_llff.py#L292"
def get_high_variance_gaussians(gaussians, viewpoints, radiis, variances, filter_ratio):
    reset_mask = [find_reset_gaussians_ratio(gaussians, viewpoints[idx], radiis[idx], variances[idx], ratio=filter_ratio) for idx in range(len(viewpoints))]

    reset_mask = torch.stack(reset_mask).float()
    reset_mask = reset_mask.sum(dim=0) > 0

    return reset_mask

def get_ens_filter_variable(scene, viewpoints, variances, radiis, quantiles, t_idx):
    high_variance_mask = get_high_variance_gaussians(scene.gaussians, viewpoints, radiis, variances, 1-quantiles[t_idx])

    threshold = torch.tensor(0.5)

    return high_variance_mask.cpu(), threshold

