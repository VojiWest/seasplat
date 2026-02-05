import torch

def get_variance(renders):
    renders = torch.tensor(renders)

    pred_mean = torch.mean(renders, dim=0)