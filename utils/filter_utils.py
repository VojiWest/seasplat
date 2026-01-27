import torch
import matplotlib.pyplot as plt

from scene import Scene, GaussianModel

def get_filter_variable(filter_criteria, gaussians : GaussianModel):
    if "sd" in filter_criteria:
        if "max" in filter_criteria:
            filter_variable = gaussians.get_sd(method='max')
        elif "mean" in filter_criteria:
            filter_variable = gaussians.get_sd(method='mean')
        return filter_variable, torch.max(filter_variable).item(), torch.min(filter_variable).item()

def filter_gaussians(filter_criteria, filter_threshold, means3D, means2D, shs, colors_precomp, opacity, scales, rotations, cov3D_precomp, remove_above_filter=True):
    if remove_above_filter:
        mask = filter_criteria > filter_threshold
    else:
        mask = filter_criteria < filter_threshold

    filtered_means3D = means3D[mask]
    filtered_means2D = means2D[mask]
    filtered_shs = shs[mask]
    filtered_colors_precomp = colors_precomp[mask]
    filtered_opacity = opacity[mask]
    filtered_scales = scales[mask]
    filtered_rotations = rotations[mask]
    filtered_cov3D_precomp = cov3D_precomp[mask]

    return filtered_means3D, filtered_means2D, filtered_shs, filtered_colors_precomp, filtered_opacity, filtered_scales, filtered_rotations, filtered_cov3D_precomp

def plot_filter(filter_criteria, filter_thresholds, l1_losses, l_ssims, psnrs, folder_path, iteration, split):
    x = filter_thresholds

    plt.plot(x, l1_losses)
    title = "Test L1 Loss Filtering" + str(filter_criteria)
    plt.title(title)
    plt.ylabel("L1 Loss")
    x_label = str(filter_criteria) + "Filtering Threshold"
    plt.xlabel(x_label)
    plt.savefig(f"{folder_path}/{split}_{filter_criteria}_loss_filter_plot_{iteration}.png")
    plt.close()

    plt.plot(x, psnrs)
    title = "Test PSNR Filtering" + str(filter_criteria)
    plt.title(title)
    plt.ylabel("PSNR")
    x_label = str(filter_criteria) + "Filtering Threshold"
    plt.xlabel(x_label)
    plt.savefig(f"{folder_path}/{split}_{filter_criteria}_psnr_filter_plot_{iteration}.png")
    plt.close()