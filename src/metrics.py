import torch
from torch.nn import CrossEntropyLoss, NLLLoss

__all__ = ["EMD"]


def EMD(prediction, ground_truth, r=2) -> torch.Tensor:
    #     print(prediction.shape, ground_truth.shape)
    mini_batch, length = ground_truth.shape[:2]
    loss_vector = []
    for pred, gt in zip(prediction, ground_truth):
        single_loss = 0.0
        for k in range(1, length + 1):
            single_loss += torch.abs(sum(pred[:k] - gt[:k])) ** r
        loss_vector.append((single_loss / length) ** (1.0 / r))

    return torch.cat(loss_vector)


nlloss = NLLLoss(reduction="none")


def NLLLoss(prediction, ground_truth) -> torch.Tensor:
    return nlloss(prediction, ground_truth).squeeze()

celoss = CrossEntropyLoss(reduction="none")


def CrossEntropyLoss(prediction, ground_truth) -> torch.Tensor:
    return celoss(prediction, ground_truth).squeeze()
