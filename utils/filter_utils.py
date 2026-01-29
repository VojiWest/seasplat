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
    # print(f"Filtered {filtered_pct} % of Gaussians with Threshold of {round(filter_threshold.item(), 5)}")

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
                sd = torch.std(gaussian_non_zero, dim=0)
                variances[idx] = sd
        else:
            variances[idx] = -1
            
    return variances

def get_depth_weighted_gradient_variance(variances, gaussians, viewpoint_camera, depth_lambda = 1):
    depths = get_depths(gaussians, viewpoint_camera)
    dw_variances = variances * (depth_lambda / (depths+0.0001))

    print("Variances: ", variances)
    print("Depths: ", depths)
    print("Depth Weighted Variances: ", dw_variances)

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

    # x = filter_thresholds
    x = quantiles

    title = "L1 Loss Filtering"
    for idx, loss in enumerate(l1_losses):
        linetype = ':'
        if idx % 2 == 1:
            linetype = '-'
        plt.plot(x, loss, label = splits[idx % 2] + " " + methods[idx // 2], linestyle=linetype)

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
        plt.plot(x, psnr, label = splits[idx % 2] + " " + methods[idx // 2], linestyle=linetype)
    title = "PSNR Filtering"
    plt.title(title)
    plt.ylabel("PSNR")
    x_label = "Percentile Kept"
    plt.xlabel(x_label)
    plt.legend()
    plt.savefig(f"{folder_path}/psnr_filter_plot_{iteration}.png")
    plt.close()