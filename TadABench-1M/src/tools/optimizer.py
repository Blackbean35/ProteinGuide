from typing import Dict

import torch.nn as nn
import torch.optim as optim
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from src import model as model_utils


class NoScheduler:
    def step(self):
        pass


class NoOptimizer:
    def __init__(self):
        pass

    def zero_grad(self):
        pass

    def step(self):
        pass


def get_optimizer(
    optimizer_type: str,
    model: nn.Module,
    lrs: Dict[str, float],
    weight_decays: Dict[str, float],
):
    param_settings = {m: [] for m in model_utils.MODULES}
    for name, params in model.named_parameters():
        other = True
        if not params.requires_grad:
            continue
        for module in model_utils.MODULES:
            if name.startswith(module):
                param_settings[module].append(params)
                other = False
                break
        if other:
            param_settings["other"].append(params)

    model_parameters = []
    for module, params in param_settings.items():
        if not params:
            continue
        print(f"Params of {module}: {len(params)}")
        model_parameters.append(
            {"params": params, "lr": lrs[module], "weight_decay": weight_decays[module]}
        )

    if optimizer_type == "SGD":
        return optim.SGD(model_parameters)
    elif optimizer_type == "Adam":
        return optim.Adam(model_parameters)
    elif optimizer_type == "AdamW":
        return optim.AdamW(model_parameters)
    elif optimizer_type == "Adagrad":
        return optim.Adagrad(model_parameters)
    elif optimizer_type == "NoOptimizer":
        return NoOptimizer()
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")


def get_scheduler(optimizer: optim.Optimizer, scheduler_kwargs: dict):
    """
    Return:
        scheduler: torch.optim.lr_scheduler._LRScheduler
        is_epoch_scheduler: bool,
            if False, scheduler.step() should be called in the training iteration
            if True, scheduler.step(loss) should be called in the training epoch
    """
    kwargs = scheduler_kwargs.copy()
    scheduler_type = kwargs.pop("type")

    if scheduler_type == "CosineAnnealingLR":
        is_epoch_scheduler = kwargs.pop("is_epoch", False)
        return get_cosine_schedule_with_warmup(optimizer, **kwargs), is_epoch_scheduler
    elif scheduler_type == "LinearAnnealingLR":
        is_epoch_scheduler = kwargs.pop("is_epoch", False)
        return get_linear_schedule_with_warmup(optimizer, **kwargs), is_epoch_scheduler
    elif scheduler_type == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs), True
    else:
        return NoScheduler(), False
