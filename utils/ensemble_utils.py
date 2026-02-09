import torch
from torchvision.utils import save_image
import os

def order_renders(renders, image_names):
    ordered_names = sorted(image_names)

    ordered_renders = []

    for model in len(renders):
        name_to_idx = {name: idx for idx, name in enumerate(image_names[model])}
        ordered = torch.stack([renders[model][name_to_idx[name]] for name in ordered_names], dim=0)

        ordered_renders.append(ordered)

    renders = torch.stack(ordered_renders, dim=0)

    return renders, ordered_names

def get_ensemble_variance(renders, image_names):
    renders, ordered_names = order_renders(renders, image_names)

    pred_mean = torch.mean(renders, dim=0)

    variance = torch.mean(renders ** 2, dim=0) - pred_mean ** 2

    return pred_mean, variance, ordered_names

def create_ens_path(model_path):
    ens_path = os.path.join(model_path, "Ensemble")
    if not os.path.exists(ens_path):
        os.makedirs(ens_path)

    return ens_path

def save_ens_uncertainty(variance, ordered_names, save_path):
    variance_norm = torch.clamp(variance / torch.max(variance), 0.0, 1.0)
    for var_image, img_name in zip(variance_norm, ordered_names):
        var_gray = var_image.mean(dim=0, keepdim=True)
        image_name = f"EnsUQ_" + {img_name} + ".png"
        save_image(var_gray, f"{save_path}/{image_name}")

def save_ens_mean_pred(pred_mean, ordered_names, save_path):
    for pred, img_name in zip(pred_mean, ordered_names):
        pred_norm = torch.clamp(pred / torch.max(pred), 0.0, 1.0)
        # var_gray = var_image.mean(dim=0, keepdim=True)
        image_name = f"EnsMean_" + {img_name} + ".png"
        save_image(pred_norm, f"{save_path}/{image_name}")
