import torch
import matplotlib.pyplot as plt

from scene import Scene, GaussianModel

def get_filter_variable(filter_criteria, gaussians : GaussianModel):
    if "sd" in filter_criteria:
        if "max" in filter_criteria:
            filter_variable = gaussians.get_sd(method='max')
        elif "mean" in filter_criteria:
            filter_variable = gaussians.get_sd(method='mean')
        return filter_variable
    if "vog" in filter_criteria:
        grads = gaussians.get_inter_view_gradients()
        if "viewpoint" in filter_criteria:
            variance = get_inter_view_gradient_variance(gradients=grads, method='sd')
            return variance

def filter_gaussians(filter_criteria, filter_threshold, means3D, means2D, shs, colors_precomp, opacity, scales, rotations, cov3D_precomp, remove_above_filter=True):
    if remove_above_filter:
        keep = filter_criteria < filter_threshold
    else:
        keep = filter_criteria > filter_threshold

    filtered_means3D = means3D[keep]
    filtered_means2D = means2D[keep]

    if shs is not None:
        filtered_shs = shs[keep]
    else:
        filtered_shs = shs

    if colors_precomp is not None:
        filtered_colors_precomp = colors_precomp[keep]
    else:
        filtered_colors_precomp = colors_precomp

    filtered_opacity = opacity[keep]
    filtered_scales = scales[keep]
    filtered_rotations = rotations[keep]
    filtered_cov3D_precomp = cov3D_precomp

    filtered_pct = round(100 - (100 * (filtered_opacity.numel() / opacity.numel())), 4)
    print(f"Filtered {filtered_pct} % of Gaussians with Threshold of {round(filter_threshold.item(), 5)}")

    return filtered_means3D, filtered_means2D, filtered_shs, filtered_colors_precomp, filtered_opacity, filtered_scales, filtered_rotations, filtered_cov3D_precomp

def get_inter_view_gradient_variance(gradients, method='var'):
    variances = torch.zeros(gradients.shape[0])
    for idx, gaussian_grads in enumerate(gradients):
        mask = gaussian_grads > 0
        gaussian_non_zero = gaussian_grads[mask]

        if gaussian_non_zero.numel() > 1:
            if method == 'var':
                var = torch.var(gaussian_non_zero, dim=0)
                variances[idx] = var
            if method == 'sd':
                sd = torch.sd(gaussian_non_zero, dim=0)
                variances[idx] = sd
        else:
            variances[idx] = -1
            
    return variances

def plot_filter(filter_criteria, filter_thresholds, quantiles, l1_losses, l_ssims, psnrs, folder_path, iteration, split):
    # x = filter_thresholds
    x = quantiles

    plt.plot(x, l1_losses)
    title = "Test L1 Loss Filtering" + str(filter_criteria)
    plt.title(title)
    plt.ylabel("L1 Loss")
    x_label = str(filter_criteria) + " Percentile Kept"
    plt.xlabel(x_label)
    plt.savefig(f"{folder_path}/{split}_{filter_criteria}_loss_filter_plot_{iteration}.png")
    plt.close()

    plt.plot(x, psnrs)
    title = "Test PSNR Filtering" + str(filter_criteria)
    plt.title(title)
    plt.ylabel("PSNR")
    x_label = str(filter_criteria) + " Percentile Kept"
    plt.xlabel(x_label)
    plt.savefig(f"{folder_path}/{split}_{filter_criteria}_psnr_filter_plot_{iteration}.png")
    plt.close()