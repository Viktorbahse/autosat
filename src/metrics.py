"""Metrics for segmentation."""

import numpy as np
import torch

NAN = float("nan")


class Metrics:
    """Tracking mean metrics"""

    def __init__(self, labels, metric):
        """Creates an new `Metrics` instance.

        Args:
          labels: the labels for all classes.
        """

        self.labels = labels
        self.num_classes = len(labels)
        self.metric = metric
        # Initialize confusion matrix
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.long)

    def add(self, predicted, actual):  # noqa: WPS210
        """Adds an observation to the tracker."""

        predicted_classes = torch.argmax(predicted, dim=1)  # shape: [N, H, W]

        predicted_flat = predicted_classes.view(-1)
        actual_flat = actual.view(-1)

        valid_mask = actual_flat >= 0
        predicted_flat = predicted_flat[valid_mask]
        actual_flat = actual_flat[valid_mask]

        valid_true = (actual_flat >= 0) & (actual_flat < self.num_classes)
        valid_pred = (predicted_flat >= 0) & (predicted_flat < self.num_classes)
        valid = valid_true & valid_pred

        actual_flat = actual_flat[valid]
        predicted_flat = predicted_flat[valid]

        if len(actual_flat) > 0:
            device = self.confusion_matrix.device
            actual_flat = actual_flat.to(device)
            predicted_flat = predicted_flat.to(device)

            idx = actual_flat * self.num_classes + predicted_flat
            counts = torch.bincount(idx, minlength=self.num_classes * self.num_classes)
            counts = counts.to(device)
            self.confusion_matrix += counts.view(self.num_classes, self.num_classes)

    def get_miou(self):  # noqa: WPS210
        """Retrieves the mean Intersection over Union score.

        Returns:
          The mean Intersection over Union score for all observations seen so far.
        """
        # Calculate IoU for each class
        ious = []
        for i in range(self.num_classes):
            # True positives: diagonal element for class i
            tp = self.confusion_matrix[i, i].item()
            # False positives: sum of column i minus tp
            fp = self.confusion_matrix[0:-1, i].sum().item() - tp
            # False negatives: sum of row i minus tp
            fn = self.confusion_matrix[i, 0:-1].sum().item() - tp

            denominator = tp + fp + fn
            if denominator > 0:
                iou = tp / denominator
            else:
                iou = NAN
            ious.append(iou)

        # Calculate mean IoU (ignoring NaN values)
        miou = np.nanmean(ious)
        return miou

    def get_fg_iou(self):
        """Retrieves the foreground Intersection over Union score.

        Returns:
          The foreground Intersection over Union score for all observations seen so far.
        """
        # Assuming class 1 is foreground (adjust if needed)
        if self.num_classes < 2:
            return NAN

        # For binary segmentation, class 1 is foreground
        tp = self.confusion_matrix[1, 1].item()
        fp = self.confusion_matrix[0:-1, 1].sum().item() - tp
        fn = self.confusion_matrix[1, 0:-1].sum().item() - tp

        denominator = tp + fp + fn
        if denominator > 0:
            iou = tp / denominator
        else:
            iou = NAN

        return iou

    def get_mcc(self):  # noqa: WPS210
        """Retrieves the Matthew's Coefficient Correlation score.
        Returns:
          The Matthew's Coefficient Correlation score for all observations seen so far.
        """
        C = self.confusion_matrix.float()
        total = C.sum()
        row_sums = C.sum(dim=1)
        col_sums = C.sum(dim=0)

        num = torch.sum(C.diag() * total - row_sums * col_sums)

        term1 = total**2 - torch.sum(row_sums**2)
        term2 = total**2 - torch.sum(col_sums**2)

        den = torch.sqrt(term1 * term2)

        mcc = NAN if den == 0 else num / den

        return mcc

    def compute(self):
        """Compute the metric for the current state."""
        if self.metric == "miou":
            return self.get_miou()
        elif self.metric == "mcc":
            return self.get_mcc()
        elif self.metric == "fg_iou":
            return self.get_fg_iou()
        return NAN

    def reset(self):
        """Reset the metrics."""
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.long)
