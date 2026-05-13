import os
import random
import argparse
import numpy as np
import importlib.util
from tqdm import tqdm
from typing import Any
from datetime import datetime

import torch
from torch.utils.data import DataLoader

import src.dataset as dataset_utils
import src.model as model_utils
from src.tools import (
    get_loss_func,
    get_optimizer,
    get_scheduler,
    test_model,
    best_eval_metric,
)


def load_args():
    parser = argparse.ArgumentParser(
        description="Load config and print seq_path variable"
    )
    parser.add_argument(
        "--cfg_path", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()
    return args


def import_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    assert spec is not None and spec.loader is not None, (
        f"Failed to load config from {config_path}"
    )
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def load_config(config_path: Any = None):
    if config_path is None:
        args = load_args()
        config_path = args.cfg_path

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    config = import_config(config_path)
    set_seed(getattr(config, "seed", 42))
    return config


def set_seed(seed: int):
    print(f"Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(head_model_type, device, cfg):
    assert hasattr(model_utils, head_model_type), (
        f"Invalid head model type: {head_model_type}"
    )
    model = getattr(model_utils, head_model_type)(cfg=cfg).to(device)
    return model


def seed_worker(worker_id):
    # Seed for each worker, not the main process
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset(cfg, has_val):
    assert hasattr(dataset_utils, cfg.dataset_type), (
        f"Invalid dataset type: {cfg.dataset_type}"
    )
    trainset = getattr(dataset_utils, cfg.dataset_type)(cfg, split="train")
    testset = getattr(dataset_utils, cfg.dataset_type)(cfg, split="test")
    if has_val:
        valset = getattr(dataset_utils, cfg.dataset_type)(cfg, split="val")

    num_workers = getattr(cfg, "num_workers", 4)
    testloader = DataLoader(
        testset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    trainloader = DataLoader(
        trainset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    if has_val:
        valloader = DataLoader(
            valset,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )
        return trainset, valset, testset, trainloader, valloader, testloader
    else:
        # if only train and test, regard valset as testset
        return trainset, testset, None, trainloader, testloader, None


def train_and_test(cfg, wandb):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    has_val = getattr(cfg, "train_val_test", False)
    use_embed_mapper = getattr(cfg, "use_embed_mapper", False)

    # if only train and test, valset = testset, testset = None
    trainset, valset, testset, trainloader, valloader, testloader = get_dataset(
        cfg, has_val
    )

    if has_val:
        wandb.config.update(
            {
                "train_num": len(trainset),
                "val_num": len(valset),
                "test_num": len(testset),
            }
        )
    else:
        wandb.config.update({"train_num": len(trainset), "test_num": len(valset)})

    model = get_model(cfg.head_model_type, device, cfg)
    loss_func = (
        get_loss_func(cfg.loss_type)
        if isinstance(cfg.loss_type, str)
        else get_loss_func(cfg.loss_type)
    )

    optimizer = get_optimizer(
        cfg.optimizer_type,
        model,
        getattr(cfg, "learning_rate", 0),
        getattr(cfg, "weight_decay", 0),
    )
    sched_cfg = getattr(cfg, "scheduler_kwargs", {"type": "NoScheduler"})
    scheduler, is_epoch_scheduler = get_scheduler(optimizer, sched_cfg)

    print("Training...")
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(cfg.save_dir, time_str)
    best_metric_val = {eval_metric: -1 for eval_metric in cfg.evaluation}
    if getattr(cfg, "eval_before_train", True):
        print("Evaluating before training...")
        val_scores, loss_eval = test_model(
            model,
            valloader,
            wandb,
            0,
            "val",
            cfg.evaluation,
            loss_func,
        )
        test_scores, loss_eval = test_model(
            model,
            testloader,
            wandb,
            0,
            "test",
            cfg.evaluation,
            loss_func,
        )
        for eval_metric in cfg.evaluation:
            best_metric_val[eval_metric] = best_eval_metric(
                eval_metric, best_metric_val[eval_metric], val_scores[eval_metric]
            )

    model.train()
    if not use_embed_mapper:
        model.backbone.model.train()
    for epoch in range(1, cfg.num_epochs + 1):
        epoch_loss = 0.0
        for data, labels in tqdm(
            trainloader, desc=f"Epoch {epoch}, Training", dynamic_ncols=True
        ):
            if not isinstance(labels, torch.Tensor):
                labels = torch.stack(labels).T

            # data/label: shape = [k, batch_size]
            optimizer.zero_grad()

            predicted_scores = model(data)

            labels = labels.to(device).to(dtype=predicted_scores.dtype)
            loss = loss_func(labels, predicted_scores)

            cfg_tmp = {"epoch": epoch, "train loss": loss.item()}
            for i, group in enumerate(optimizer.param_groups):
                cfg_tmp[f"lr_{i}"] = group["lr"]
            wandb.log(cfg_tmp)

            loss.backward()
            optimizer.step()
            if not is_epoch_scheduler:
                scheduler.step()
            epoch_loss += loss.item()

        print(
            f"\nEpoch {epoch}/{cfg.num_epochs}, Train Loss: {epoch_loss / len(trainloader)}"
        )

        if epoch % cfg.eval_interval == 0 or (epoch == cfg.num_epochs):
            scores, loss_eval = test_model(
                model,
                valloader,
                wandb,
                epoch,
                "val",
                cfg.evaluation,
                loss_func,
            )
            for eval_metric in cfg.evaluation:
                best_metric_val[eval_metric] = best_eval_metric(
                    eval_metric, best_metric_val[eval_metric], scores[eval_metric]
                )
            if is_epoch_scheduler:
                if sched_cfg["type"] == "ReduceLROnPlateau":
                    scheduler.step(loss_eval)
                else:
                    scheduler.step()

            model.train()
            if not use_embed_mapper:
                model.backbone.model.train()

        if cfg.save_interval > -1 and (
            epoch % cfg.save_interval == 0 or (epoch == cfg.num_epochs)
        ):
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch}.pth"))

    if getattr(cfg, "eval_after_train", True) and testloader is not None:
        scores = test_model(
            model,
            testloader,
            wandb,
            epoch,
            "test",
            cfg.evaluation,
            loss_func,
        )

    wandb.log(
        {
            f"best {eval_metric} score": best_metric_val[eval_metric]
            for eval_metric in cfg.evaluation
        }
    )
