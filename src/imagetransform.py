from torchvision import transforms


class ImageTransform(object):
    def __init__(
        self,
        resize_size,
        crop_size,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        interpolation=None,
        center_crop: bool = False,
    ) -> None:
        self.resize_transform = transforms.Resize(
            resize_size, interpolation=interpolation
        )
        self.crop_transform = (
            transforms.CenterCrop(crop_size)
            if center_crop
            else transforms.RandomCrop(crop_size)
        )
        self.normalize_transform = transforms.Normalize(mean=mean, std=std)
        self._info = f"ImageTransform( resize_size = {resize_size}, crop_size = {crop_size}, mean = {mean}, std = {std}, interpolation = {interpolation} )"

    def __str__(self) -> str:
        return self._info

    @property
    def TrainTransform(self):
        return transforms.Compose(
            [
                self.resize_transform,
                self.crop_transform,
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    @property
    def TestTransform(self):
        return transforms.Compose(
            [
                self.resize_transform,
                self.crop_transform,
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )
