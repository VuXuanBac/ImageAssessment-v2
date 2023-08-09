import pandas as pd
import torch
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import os


class AVADataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        annotation_file: str,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.root = images_dir
        self.info = pd.read_csv(annotation_file, header=None)

    def __getitem__(self, index: int) -> tuple[Image.Image, torch.tensor, int]:
        item_info = self.info.iloc[index]
        id = int(item_info[0])

        image_path = os.path.join(self.root, f"{id}.jpg")

        image = Image.open(image_path).convert("RGB")
        label = torch.tensor(item_info[1:].values, dtype=torch.float).view(-1, 1)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return id, image, label

    def __len__(self):
        return len(self.info)


def create_ava_dataloader(dataloader_info: dict):
    """
    [dataloader_infos]: Dictionary data used to create dataloaders.
        - 'images_dir':       <path to directory contains images>,
        - 'annotation_file':  <path to annotation file for labels>,
        - 'batch_size':       <number of samples per batch>,
        - 'shuffle':          <shuffle when loading dataset or not>,
        - 'image_transform':  [Optional, None] <transform used for image, applied when loading>,
        - 'label_transform':  [Optional, None] <transform used for label, applied when loading>,
        - 'num_workers':      [Optional, 0] <number of workers used for parallel-loading>
    """
    images_dir = dataloader_info["images_dir"]
    annotation_file = dataloader_info["annotation_file"]
    batch_size = dataloader_info["batch_size"]
    shuffle = dataloader_info["shuffle"]
    image_transform = dataloader_info.get("image_transform", None)
    label_transform = dataloader_info.get("label_transform", None)
    num_workers = dataloader_info.get("num_workers", 0)

    dataset = AVADataset(images_dir, annotation_file, image_transform, label_transform)
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)


class MergeDataset(Dataset):
    def __init__(
        self,
        input_anno_files: list[str],
        label_anno_file: str = None,
        merge_on: str = "id",
    ) -> None:
        super().__init__()
        dfs = [pd.read_csv(inp, header=None) for inp in input_anno_files]

        merged = dfs[0]
        for i in range(1, len(dfs)):
            merged = pd.merge(merged, dfs[i], on=merge_on)
        # print('Merge shape', merged.shape)
        self.inputs = merged
        self.labels = pd.read_csv(label_anno_file, header=None)

    def __getitem__(self, index):
        id = int(self.labels.iloc[index][0])
        inp = torch.tensor(self.inputs.iloc[index][1:].values, dtype=torch.float)
        lbl = torch.tensor(self.labels.iloc[index][1:].values, dtype=torch.float).view(
            -1, 1
        )
        return id, inp, lbl

    def __len__(self):
        return len(self.inputs)


def create_merge_dataloader(
    input_anno_files: list[str],
    batch_size: int,
    merge_on: str = "id",
    label_anno_file: str = None,
    shuffle=True,
    num_workers: int = 0,
):
    dset = MergeDataset(
        input_anno_files,
        label_anno_file=label_anno_file,
        merge_on=merge_on,
    )
    return DataLoader(dset, batch_size, shuffle=shuffle, num_workers=num_workers)
