import torch
import numpy as np
import os
import pandas as pd

from torch import nn
import datetime


__all__ = [
    "default_device",
    "set_checkpoint",
    "write_state_on_file",
    "get_parameters_info",
    "unfreeze",
    "join_path",
    "optimizer_info",
    "print_current_time",
    "fixed_random",
    "concat_save",
    "OneHotTransform",
    "LabelTransform",
]


def LabelTransform(possibilities, splits=[0, 4.5, 6.5, 10]):
    score = (
        (torch.arange(1, len(possibilities) + 1).view(-1, 1) * possibilities)
        .sum()
        .item()
    )
    # print(score, torch.arange(1, len(possibilities) + 1).view(-1, 1) * possibilities)
    for i in range(1, len(splits)):
        if score <= splits[i]:
            return torch.tensor([i - 1])
    raise Exception("Class out of ranges")


def OneHotTransform(possibilities, splits=[0, 4.5, 6.5, 10]):
    score = (
        (torch.arange(1, len(possibilities) + 1).view(-1, 1) * possibilities)
        .sum()
        .item()
    )
    # print(score, torch.arange(1, len(possibilities) + 1).view(-1, 1) * possibilities)
    res = torch.zeros(len(splits) - 1, 1)
    for i in range(1, len(splits)):
        if score <= splits[i]:
            res[i - 1] = 1.0
            break
    return res


def fixed_random(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def concat_save(
    input_files: list[str],
    output_file: str,
    *,
    header=None,
    dtypes: dict = None,
    column_cut=slice(None, None),
    row_cut=slice(None, None),
):
    concated = pd.concat(
        [pd.read_csv(inp, header=header) for inp in input_files], ignore_index=True
    )
    if dtypes:
        concated = concated.astype(dtypes)
    concated.loc[row_cut, column_cut].to_csv(output_file, header=header, index=None)
    return concated


def print_current_time(print_before="====== Current Time:", print_after="======"):
    print(print_before, datetime.datetime.now(), print_after)


def optimizer_info(optimizer):
    name = str(optimizer.__class__)
    name = name[name.rfind(".") + 1 : -2]
    res = []
    for ind, pg in enumerate(optimizer.param_groups):
        init = f" / {pg['initial_lr']:.2e}" if "initial_lr" in pg else ""
        res.append(f"group_{ind}: {pg['lr']:.2e}{init}")
    return f"{name} ( " + ", ".join(res) + " )"


def join_path(base, *others):
    return os.path.join(base, *others)


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_checkpoint(epoch, model, optimizer, checkpoint_dir: str, **kwargs):
    path_to_checkpoint = os.path.join(checkpoint_dir, f"epoch-{epoch}.tar")
    data = {"model": model.state_dict(), "optim": optimizer.state_dict()}
    data.update(kwargs)
    torch.save(data, path_to_checkpoint)


def write_state_on_file(path, *values, delimiter=","):
    with open(path, "a") as file:
        file.write(delimiter.join(str(v) for v in values))
        file.write("\n")


def get_parameters_info(model: nn.Module):
    trainable, untrainable = 0, 0
    for param in model.parameters():
        if param.requires_grad:
            trainable += param.numel()
        else:
            untrainable += param.numel()
    return trainable, untrainable


def unfreeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True
