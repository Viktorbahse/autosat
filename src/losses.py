"""PyTorch-compatible losses and loss functions."""

import torch


class CrossEntropyLoss2d(torch.nn.Module):
    """Cross-entropy.

    See: http://cs231n.github.io/neural-networks-2/#losses
    """

    def __init__(self, weight=None):
        """Creates an `CrossEntropyLoss2d` instance.

        Args:
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = torch.nn.NLLLoss(weight)

    def forward(self, inputs, targets):
        return self.nll_loss(torch.nn.functional.log_softmax(inputs, dim=1), targets)


class FocalLoss2d(torch.nn.Module):
    """Focal Loss.

    Reduces loss for well-classified samples putting focus on hard mis-classified samples.

    See: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, gamma=2, weight=None):
        """Creates a `FocalLoss2d` instance.

        Args:
          gamma: the focusing parameter; if zero this loss is equivalent with `CrossEntropyLoss2d`.
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = torch.nn.NLLLoss(weight)
        self.gamma = gamma

    def forward(self, inputs, targets):
        penalty = (1 - torch.nn.functional.softmax(inputs, dim=1)) ** self.gamma
        return self.nll_loss(penalty * torch.nn.functional.log_softmax(inputs, dim=1), targets)


class mIoULoss2d(torch.nn.Module):
    """mIoU Loss.

    See:
      - http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
      - http://www.cs.toronto.edu/~wenjie/papers/iccv17/mattyus_etal_iccv17.pdf
    """

    def __init__(self, weight=None):
        """Creates a `mIoULoss2d` instance.

        Args:
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = torch.nn.NLLLoss(weight)

    def forward(self, inputs, targets):  # noqa: WPS210
        N, C, H, W = inputs.size()

        # Ensure targets are int64 for scatter operation
        targets = targets.long()  # Add this line to ensure int64 dtype

        softs = torch.nn.functional.softmax(inputs, dim=1).permute(1, 0, 2, 3)
        masks = torch.zeros(N, C, H, W, device=targets.device)
        masks.scatter_(1, targets.view(N, 1, H, W), 1)
        masks = masks.permute(1, 0, 2, 3)

        inters = softs * masks
        unions = (softs + masks) - (softs * masks)

        inters_flat = inters.view(C, N, -1).sum(2)
        unions_flat = unions.view(C, N, -1).sum(2)
        miou = 1.0 - (inters_flat / unions_flat).mean()

        return max(miou, self.nll_loss(torch.nn.functional.log_softmax(inputs, dim=1), targets))


class LovaszLoss2d(torch.nn.Module):
    """Lovasz Loss.

    See: https://arxiv.org/abs/1705.08790
    """

    def forward(self, inputs, targets):  # noqa: WPS210

        N, C, H, W = inputs.size()
        masks = torch.zeros(N, C, H, W).to(targets.device)
        masks = masks.scatter_(1, targets.view(N, 1, H, W), 1)

        loss = float(0)

        for mask, input in zip(masks.view(N, -1), inputs.view(N, -1)):
            max_margin_errors = 1.0 - ((mask * 2 - 1) * input)
            errors_sorted, indices = torch.sort(max_margin_errors, descending=True)
            labels_sorted = mask[indices.data]

            inter = labels_sorted.sum() - labels_sorted.cumsum(0)
            union = labels_sorted.sum() + (1.0 - labels_sorted).cumsum(0)
            iou = 1.0 - inter / union

            p = len(labels_sorted)
            if p > 1:
                iou[1:p] = iou[1:p] - iou[0:-1]  # noqa: WPS362 WPS349

            loss += torch.dot(torch.nn.functional.relu(errors_sorted), iou)

        return loss / N
