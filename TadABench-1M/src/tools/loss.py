import torch.nn.functional as F
from typing import Union


def mse_loss(y_true, y_pred):
    y_true = y_true.float()
    y_pred = y_pred.float()

    loss = F.mse_loss(y_pred, y_true)
    return loss


def get_loss_func(loss_info: Union[str, dict]):
    if isinstance(loss_info, str):
        loss_info = {"type": loss_info}

    if loss_info["type"] == "mse":
        return mse_loss
    else:
        raise ValueError(f"Invalid loss type: {loss_info}")
