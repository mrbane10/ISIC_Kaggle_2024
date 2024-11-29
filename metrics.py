from sklearn.metrics import roc_curve, auc
from torch import Tensor
from torchmetrics.classification import BinaryPrecisionRecallCurve as _BinaryPrecisionRecallCurve
from torchmetrics.functional.classification.precision_recall_curve import (
    _adjust_threshold_arg,
    _binary_precision_recall_curve_update,
)
import torch
import numpy as np
from torchmetrics import ROC, Metric
from torchmetrics.functional import auc as auc_

def normalize(anomaly_map, min_val, max_val, threshold = 0.5):
    anomaly_map = torch.tensor(anomaly_map)
    normalized = ((anomaly_map - threshold) / (max_val - min_val))
    normalized = torch.minimum(normalized, torch.tensor(1.))
    normalized = torch.maximum(normalized, torch.tensor(0.))
    return normalized

class BinaryPrecisionRecallCurve(_BinaryPrecisionRecallCurve):
    """Binary precision-recall curve with without threshold prediction normalization."""
    @staticmethod
    def _binary_precision_recall_curve_format(
        preds: Tensor,
        target: Tensor,
        thresholds: int | list[float] | Tensor | None = None,
        ignore_index: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Similar to torchmetrics' ``_binary_precision_recall_curve_format`` except it does not apply sigmoid."""
        preds = preds.flatten()
        target = target.flatten()
        if ignore_index is not None:
            idx = target != ignore_index
            preds = preds[idx]
            target = target[idx]

        thresholds = _adjust_threshold_arg(thresholds, preds.device)
        return preds, target, thresholds

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state with new predictions and targets.

        Unlike the base class, this accepts raw predictions and targets.

        Args:
            preds (Tensor): Predicted probabilities
            target (Tensor): Ground truth labels
        """
        preds, target, _ = BinaryPrecisionRecallCurve._binary_precision_recall_curve_format(
            preds,
            target,
            self.thresholds,
            self.ignore_index,
        )
        state = _binary_precision_recall_curve_update(preds, target, self.thresholds)
        if isinstance(state, Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])


class F1AdaptiveThreshold(BinaryPrecisionRecallCurve):
    def __init__(self, default_value: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_state("value", default=torch.tensor(default_value), persistent=True)
        self.value = torch.tensor(default_value)

    def compute(self) -> torch.Tensor:

        precision: torch.Tensor
        recall: torch.Tensor
        thresholds: torch.Tensor

        precision, recall, thresholds = super().compute()
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
        if thresholds.dim() == 0:
            # special case where recall is 1.0 even for the highest threshold.
            # In this case 'thresholds' will be scalar.
            self.value = thresholds
        else:
            self.value = thresholds[torch.argmax(f1_score)]
        return self.value


class MinMax(Metric):
    full_state_update: bool = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("min", torch.tensor(float("inf")), persistent=True)
        self.add_state("max", torch.tensor(float("-inf")), persistent=True)

        self.min = torch.tensor(float("inf"))
        self.max = torch.tensor(float("-inf"))

    def update(self, predictions: torch.Tensor, *args, **kwargs) -> None:
        del args, kwargs  # These variables are not used.

        self.max = torch.max(self.max, torch.max(predictions))
        self.min = torch.min(self.min, torch.min(predictions))

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.min, self.max


class Partial_AUROC(Metric):
    def __init__(self):
        super().__init__()
        self.preds = []
        self.labels = []

    def update(self, pred, label):
        self.preds.append(pred)
        self.labels.append(label)

    def compute(self):
        preds = torch.cat(self.preds, dim = 0)
        labels = torch.cat(self.labels, dim=0)
        return compute_pauc(labels, preds)

    def reset(self):
        self.preds = []
        self.labels = []


def compute_pauc(y_true, y_scores, tpr_threshold=0.8):
    """
    Compute the partial AUC above a given TPR threshold.

    Parameters:
    y_true (np.array): True binary labels.
    y_scores (np.array): Target scores.
    tpr_threshold (float): TPR threshold above which to compute the pAUC.
    Returns:
    float: The partial AUC above the given TPR threshold.
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Find the indices where the TPR is above the threshold
    tpr_above_threshold_indices = np.where(tpr >= tpr_threshold)[0]

    if len(tpr_above_threshold_indices) == 0:
        return 0.0

    # Extract the indices for the ROC segment above the threshold
    start_index = tpr_above_threshold_indices[0]
    fpr_above_threshold = fpr[start_index:]
    tpr_above_threshold = tpr[start_index:] - tpr_threshold

    partial_auc = auc(fpr_above_threshold, tpr_above_threshold)
    return partial_auc


class AUROC(ROC):
    def __init__(self):
        super().__init__()

    def compute(self) -> torch.Tensor:
        tpr: torch.Tensor
        fpr: torch.Tensor
        fpr, tpr = self._compute()
        return auc_(fpr, tpr, reorder=True)
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        super().update(preds.flatten(), target.flatten())
    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        tpr: torch.Tensor
        fpr: torch.Tensor
        fpr, tpr, _thresholds = super().compute()
        return (fpr, tpr)