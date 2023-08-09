from torch import nn
from torchvision import models


class FeatureLearner(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        n_input: int,
        n_output: int,
        *,
        dropout,
        freeze_base_weights
    ) -> None:
        super(FeatureLearner, self).__init__()

        self.features = features

        if freeze_base_weights:
            for param in self.features.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(n_input, n_output), nn.Softmax(dim=-1)
        )

        self.n_classes = n_output

    def forward(self, x):
        f_pred = self.features(x)
        f_pred = f_pred.view(f_pred.size(0), -1)
        return self.classifier(f_pred)


def create_feature_learner(
    name: str, pretrained_weights: str = "DEFAULT", *, n_output: int = 10
):
    shortnames = {
        "eff4": "efficientnet_b4",
        "eff3": "efficientnet_b3",
        "effv2": "effcientnet_v2_s",
        "conv": "convnext_tiny",
        "regx": "regnet_x_3_2_gf",
        "regy": "regnet_y_1_6gf",
        "swin": "swin_v2_t",
    }
    if name in shortnames:
        name = shortnames[name]
    try:
        base_model = models.get_model(name, weights=pretrained_weights)
    except:
        print("You can use supported shortname", shortnames)
        raise

    if name.startswith("eff"):
        features = nn.Sequential(base_model.features, base_model.avgpool)
        inp = base_model.classifier[1].in_features
    elif name.startswith("reg"):
        features = nn.Sequential(
            base_model.stem, base_model.trunk_output, base_model.avgpool
        )
        inp = base_model.fc.in_features
    elif name.startswith("conv"):
        features = nn.Sequential(base_model.features, base_model.avgpool)
        inp = base_model.classifier[2].in_features
    elif name.startswith("swin"):
        features = nn.Sequential(
            base_model.features, base_model.norm, base_model.permute, base_model.avgpool
        )
        inp = base_model.head.in_features
    else:
        raise NotImplementedError("Not support model name", name)

    weight = models.get_model_weights(name)[pretrained_weights]
    used_tf = weight.transforms()
    used_transform = {
        "resize_size": used_tf.resize_size,
        "crop_size": used_tf.crop_size,
        "mean": used_tf.mean,
        "std": used_tf.std,
        "interpolation": used_tf.interpolation,
    }
    print(":: Loaded Model", name.upper(), "with Pretrained Weights", weight.name)
    return (
        FeatureLearner(features, inp, n_output, dropout=0.75, freeze_base_weights=True),
        used_transform,
    )
