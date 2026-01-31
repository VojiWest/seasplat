import torch
import os
import matplotlib.pyplot as plt

from scene import Scene, GaussianModel

def get_filter_variable(filter_criteria, gaussians : GaussianModel, model_path, iteration):
    if "sd" in filter_criteria:
        if "max" in filter_criteria:
            filter_variable = gaussians.get_sd(method='max')
        elif "mean" in filter_criteria:
            filter_variable = gaussians.get_sd(method='mean')
        return filter_variable
    if "vog" in filter_criteria:
        grads = gaussians.get_inter_view_gradients()
        if "viewpoint" in filter_criteria:
            variance = get_inter_view_gradient_variance(gradients=grads, method='sd', model_path=model_path, iteration=iteration)
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
    # print(f"Filtered {filtered_pct} % of Gaussians with Threshold of {round(filter_threshold.item(), 5)}")

    return filtered_means3D, filtered_means2D, filtered_shs, filtered_colors_precomp, filtered_opacity, filtered_scales, filtered_rotations, filtered_cov3D_precomp

def calculate_spread(gaussian_non_zero, method):
    if gaussian_non_zero.numel() > 1:
        if method == 'var':
            var = torch.var(gaussian_non_zero, dim=0)
            spread = var
        elif method == 'sd':
            sd = torch.std(gaussian_non_zero, dim=0)
            spread = sd
    elif gaussian_non_zero.numel() > 0:
        spread = 0
    else:
        spread = -1

    return spread

def get_inter_view_gradient_variance(gradients, method='var', model_path="", iteration=0):
    spreads = torch.zeros(gradients.shape[0])
    numels = []
    for idx, gaussian_grads in enumerate(gradients):
        if gaussian_grads.ndim == 2:
            g1, g2 = gaussian_grads
            g1_non_zero = g1[g1 > 0]
            g2_non_zero = g2[g2 > 0]

            numels.append(g1_non_zero.numel())
            numels.append(g2_non_zero.numel())

            v1 = calculate_spread(g1_non_zero, method=method)
            v2 = calculate_spread(g2_non_zero, method=method)

            spreads[idx] = torch.mean(torch.stack([v1, v2]))
        
        elif gaussian_grads.ndim == 1:
            mask = gaussian_grads > 0
            gaussian_non_zero = gaussian_grads[mask]

            numels.append(gaussian_non_zero.numel())

            spreads[idx] = calculate_spread(gaussian_non_zero, method=method)

    hist_path = os.path.join(model_path, "Histogram")
    plot_histogram(numels, title="Number of Gradient Samples per Gaussian", folder_path=hist_path, iteration=iteration)
            
    return spreads

def get_depth_weighted_gradient_variance(variances, gaussians, viewpoint_camera, depth_lambda = 1):
    depths = get_depths(gaussians, viewpoint_camera)
    dw_variances = variances * (depth_lambda / (depths+0.0001))

    # print("Variances: ", variances)
    # print("Depths: ", depths)
    # print("Depth Weighted Variances: ", dw_variances)

    return dw_variances

def homogenize_points(points):
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

def get_depths(gaussians, viewpoint_camera):
    # Taken from gaussian_rendeder

    points_world = gaussians.get_xyz # torch.Size([25837, 3]), torch.float32
    T_cam_world = viewpoint_camera.world_view_transform.T # torch.Size([4, 4]), torch.float32

    # homogenize gaussian centers
    points_world_homogenized = homogenize_points(points_world)

    # apply T_cam_world to these points
    #T_cam_world = T_world_cam.inverse()
    points_cam_homogenized = (T_cam_world @ points_world_homogenized.T).T
    points_cam = points_cam_homogenized[:, :3]

    # get the zs and use as color for rendering
    zs = points_cam[:, 2]

    return zs.cpu()

def plot_filter(filter_thresholds, quantiles, l1_losses, l_ssims, psnrs, folder_path, iteration, methods, split_names):
    splits = [split_names[0]['name'].split("_")[1], split_names[1]['name'].split("_")[1]]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # x = filter_thresholds
    x = quantiles

    title = "L1 Loss Filtering"
    for idx, loss in enumerate(l1_losses):
        linetype = ':'
        if idx % 2 == 1:
            linetype = '-'
        plt.plot(x, loss, color=colors[idx // 2], label = splits[idx % 2] + " " + methods[idx // 2], linestyle=linetype)

    plt.title(title)
    plt.ylabel("L1 Loss")
    x_label = "Percentile Kept"
    plt.xlabel(x_label)
    plt.legend()
    plt.savefig(f"{folder_path}/loss_filter_plot_{iteration}.png")
    plt.close()

    for idx, psnr in enumerate(psnrs):
        linetype = ':'
        if idx % 2 == 1:
            linetype = '-'
        plt.plot(x, psnr, color=colors[idx // 2], label = splits[idx % 2] + " " + methods[idx // 2], linestyle=linetype)
    title = "PSNR Filtering"
    plt.title(title)
    plt.ylabel("PSNR")
    x_label = "Percentile Kept"
    plt.xlabel(x_label)
    plt.legend()
    plt.savefig(f"{folder_path}/psnr_filter_plot_{iteration}.png")
    plt.close()

def plot_histogram(data, title, folder_path, iteration):
    plt.hist(data, bins=100)
    plt.yscale('log', nonpositive='clip')
    plt.title(title)
    plt.savefig(f"{folder_path}/hist_{title}_{iteration}.png")
    plt.close()