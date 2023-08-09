import torch
from torch import nn
import pandas as pd
import numpy as np
import os


def predict_apply(model, dataloader, *funcs, device=None):
    if isinstance(model, nn.Module):
        model.to(device)
        model.eval()
    else:
        print("...Making predict with not nn.Module model...")
    rets = [[] for _ in funcs]
    with torch.no_grad():
        for ids, inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).view(len(labels), -1, 1)

            for i in range(len(funcs)):
                x = funcs[i](outputs, labels)
                rets[i].append(x.view(-1))

    return tuple(torch.cat(ret, dim=0) for ret in rets)


def predict(model, dataloader, device=None):
    if isinstance(model, nn.Module):
        model.to(device)
        model.eval()
    else:
        print("...Making predict with not nn.Module model...")

    print(":: Predict with", len(dataloader.dataset), "samples")
    rets = {"id": [], "output": [], "label": []}
    with torch.no_grad():
        for ids, inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs).view(len(labels), -1, 1)
            rets["id"].append(ids)
            rets["label"].append(labels)
            rets["output"].append(outputs)
    return {k: torch.cat(v, dim=0).cpu() for k, v in rets.items()}


class Tester(object):
    def __init__(self, model, checkpoint_file) -> None:
        if checkpoint_file is not None and os.path.exists(checkpoint_file):
            data = torch.load(checkpoint_file)
            if "model" in data:
                model.load_state_dict(data["model"])
            self.model = model
        else:
            raise Exception(f'...Invalid checkpoint file in path {checkpoint_file}...')

    def predict_save(
        self, dataloader, prediction_file: str, *, save_label: bool = False, device=None
    ):
        pred_dir = os.path.dirname(prediction_file)
        if not os.path.exists(pred_dir):
            print(f"...Prediction path {pred_dir} does not exist...")
            os.makedirs(pred_dir)

        data = predict(self.model, dataloader, device=device)
        merge_data = [data["id"].view(-1, 1), data["output"].squeeze()]
        if save_label:
            merge_data.append(data["label"])#.squeeze())
        #print(merge_data[0], merge_data[1], merge_data[2])
        saved_data = torch.cat(merge_data, dim=1)
        with open(prediction_file, "w") as file:
            for row in saved_data:
                file.write(",".join([str(x.item()) for x in row]))
                file.write("\n")
        return saved_data
        # df = pd.DataFrame(torch.cat(merge_data, dim=1))
        # # df = df.astype({0: int})
        # print(df.head())
        # df.to_csv(prediction_file, header=None, index=None)
        # return df
