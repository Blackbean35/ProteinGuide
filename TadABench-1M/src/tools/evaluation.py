import math
import torch
import numpy as np
from typing import List
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import (
    ndcg_score as get_ndcg_ranking_score,
    roc_auc_score,
    f1_score,
    log_loss,
)


def best_eval_metric(eval_metric: str, best_metric: float, new_metric: float) -> float:
    if eval_metric in ["mse", "ece", "nll", "mae", "rmse", "mape"]:
        return min(1000 if best_metric < 0 else best_metric, new_metric)
    elif eval_metric in [
        "mrr",
        "ndcg",
        "sp",
        "erank",
        "mrr_ranking",
        "ndcg_ranking",
        "sp_ranking",
        "auroc",
        "acc",
        "f1",
        "explained_variance",
        "r2",
        "pearson",
        "recall_at_10pct",
        "ndcg_at_10pct",
    ]:
        return max(best_metric, new_metric)
    elif eval_metric in ["per_class_precision", "per_class_recall"]:
        sum_new = sum(new_metric.values())
        sum_best = (
            sum(best_metric.values()) if isinstance(best_metric, dict) else best_metric
        )
        return new_metric if sum_new > sum_best else best_metric
    else:
        raise ValueError(f"Unknown evaluation metric: {eval_metric}")


def get_mrr_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    index_in_labels = np.argmax(labels)
    predicted_value = predicted[index_in_labels]
    predicted_rank = 0

    for i in range(len(predicted)):
        # The higher the predicted score, the more top the ranking is
        if predicted[i] >= predicted_value:
            predicted_rank += 1

    return 1 / predicted_rank


def get_mrr_ranking_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    batch_size = predicted.shape[0]
    total_mrr = 0.0

    for i in range(batch_size):
        total_mrr += get_mrr_score(labels[i], predicted[i])

    return total_mrr / batch_size


def get_acc_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    predicted = np.argmax(predicted, axis=1)
    acc = np.mean(labels == predicted)
    return acc


def get_per_class_precision_score(labels: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Compute per-class precision.

    Precision = TP / (TP + FP)

    Args:
        labels: np.ndarray of shape (N,), true class labels.
        predicted: np.ndarray of shape (N, C), predicted logits or probabilities.

    Returns:
        Dictionary mapping class index to accuracy.
    """
    predicted_classes = np.argmax(predicted, axis=1)
    unique_classes = np.unique(labels)
    per_class_precision = {}

    for cls in unique_classes:
        cls_mask = predicted_classes == cls
        cls_total = np.sum(cls_mask)
        cls_correct = np.sum(labels[cls_mask] == cls)
        precision = cls_correct / cls_total if cls_total > 0 else 0.0
        per_class_precision[int(cls)] = precision

    return per_class_precision


def get_recall_at_10pct_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    n = len(labels)
    k = max(1, int(n * 0.1))  # top 10% 样本数

    # 真实 top 10% positive 定义
    label_threshold = np.percentile(labels, 90)
    positive_idx = np.where(labels >= label_threshold)[0]

    # 按 predicted 排序，取 top-k 样本
    order = np.argsort(predicted)[::-1]
    topk_idx = order[:k]

    # top-k里命中了多少positive
    true_positives = np.intersect1d(topk_idx, positive_idx).size

    if positive_idx.size == 0:
        return 0.0
    return true_positives / positive_idx.size


def get_per_class_recall_score(labels: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Compute per-class recall.

    Recall = TP / (TP + FN)

    Args:
        labels: np.ndarray of shape (N,), true class labels.
        predicted: np.ndarray of shape (N, C), predicted logits or probabilities.

    Returns:
        Dictionary mapping class index to recall.
    """
    predicted_classes = np.argmax(predicted, axis=1)
    unique_classes = np.unique(labels)
    per_class_recall = {}

    for cls in unique_classes:
        cls_mask = labels == cls
        cls_total = np.sum(cls_mask)
        cls_tp = np.sum(predicted_classes[cls_mask] == cls)
        recall = cls_tp / cls_total if cls_total > 0 else 0.0
        per_class_recall[int(cls)] = recall

    return per_class_recall


def get_ece_score(labels: np.ndarray, predicted: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE)

    Args:
        labels (np.ndarray): True labels, shape (N,)
        predicted (np.ndarray): Predicted probabilities, shape (N, C)
        n_bins (int): Number of bins to use for calibration

    Returns:
        float: Expected Calibration Error
    """
    confidences = np.max(predicted, axis=1)
    predictions = np.argmax(predicted, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin

    return ece


def get_f1_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute macro-averaged F1 score for multi-class classification.

    Args:
        labels (np.ndarray): True labels, shape (N,)
        predicted (np.ndarray): Predicted probabilities, shape (N, C)

    Returns:
        float: macro F1-score
    """
    predictions = np.argmax(predicted, axis=1)
    return f1_score(labels, predictions, average="macro")


def get_nll_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute Negative Log Likelihood (NLL) via log loss.

    Args:
        labels (np.ndarray): True labels, shape (N,)
        predicted (np.ndarray): Predicted probabilities, shape (N, C)

    Returns:
        float: Negative Log Likelihood
    """
    return log_loss(labels, predicted)


def get_ndcg_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    return get_ndcg_ranking_score([labels], [predicted])


def get_ndcg_at_10pct_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    n = len(labels)
    k = max(1, int(n * 0.1))  # top 10% 样本数，至少是1个

    # min-max normalize labels（作为 continuous gain）
    gains = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)

    # 按 predicted 排序，取前k个样本
    order = np.argsort(predicted)[::-1]
    topk_idx = order[:k]

    dcg = np.sum(gains[topk_idx] / np.log2(np.arange(2, k + 2)))

    # 理想排序（按真实 gains 排）
    ideal_order = np.argsort(gains)[::-1]
    ideal_topk_idx = ideal_order[:k]
    idcg = np.sum(gains[ideal_topk_idx] / np.log2(np.arange(2, k + 2)))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def get_mse_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    return np.mean((labels - predicted) ** 2)


def get_rmse_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    mse = get_mse_score(labels, predicted)
    return np.sqrt(mse)


def get_mae_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    return np.mean(np.abs(labels - predicted))


def get_r2_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    total_variance = np.var(labels)
    residual_variance = np.var(labels - predicted)
    return 1 - (residual_variance / total_variance)


def get_medae_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    return np.median(np.abs(labels - predicted))


def get_mape_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    epsilon = 1e-10
    return np.mean(np.abs((labels - predicted) / (labels + epsilon))) * 100


def get_explained_variance_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    variance_labels = np.var(labels)
    variance_residuals = np.var(labels - predicted)
    return 1 - (variance_residuals / variance_labels)


def get_pearson_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    pearson = np.corrcoef(labels, predicted)[0, 1]
    if math.isnan(pearson) or np.isnan(pearson):
        pearson = np.float64(0.0)
    return pearson


def get_sp_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    spc, p = spearmanr(labels, predicted)
    if math.isnan(spc) or np.isnan(spc):
        spc = np.float64(0.0)
    return spc


def get_auroc_score(labels: np.ndarray, predicted: np.ndarray) -> float:
    return roc_auc_score(labels, predicted)


def get_sp_ranking_score(
    labels_batch: np.ndarray, predicted_batch: np.ndarray
) -> float:
    batch_size = predicted_batch.shape[0]
    total_sp = 0.0

    for i in range(batch_size):
        spc, p = spearmanr(labels_batch[i], predicted_batch[i])
        if math.isnan(spc) or np.isnan(spc):
            spc = np.float64(0.0)
        total_sp += spc

    return total_sp / batch_size


def test_model(
    model,
    dataloader,
    wandb,
    epoch: int,
    mode: str,
    evaluation: List[str],
    loss_func,
):
    if not isinstance(evaluation, list):
        evaluation = [evaluation]

    model.eval()
    preds_all, labels_all = [], []
    for test_data, test_labels in tqdm(
        dataloader, desc=f"Epoch {epoch}, {mode}ing", dynamic_ncols=True
    ):
        with torch.no_grad():
            predicted_scores = (
                model(test_data)
                .to("cpu")
                .softmax(dim=-1)
                .to(dtype=torch.float32)
                .numpy()
            )
            if not isinstance(test_labels, torch.Tensor):
                labels = torch.stack(test_labels).T.to("cpu").numpy()
            else:
                labels = test_labels.to("cpu").numpy()
        preds_all.append(predicted_scores)
        labels_all.append(labels)

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    total_score = {}
    for eval_metric in evaluation:
        total_score[eval_metric] = globals()[f"get_{eval_metric}_score"](
            labels_all, preds_all
        )

    loss = loss_func(torch.tensor(labels_all), torch.tensor(preds_all))

    total_score = {
        eval_metric: total_score[eval_metric]
        if isinstance(total_score[eval_metric], dict)
        else float(total_score[eval_metric])
        for eval_metric in evaluation
    }

    print(f"Epoch {epoch}, {mode}ing {len(labels_all)} samples: {total_score}")
    wandb.log(
        {
            "epoch": epoch,
            f"{mode}_loss": loss,
            **{
                f"{mode}_{eval_metric}": total_score[eval_metric]
                for eval_metric in evaluation
            },
        }
    )

    return total_score, loss
