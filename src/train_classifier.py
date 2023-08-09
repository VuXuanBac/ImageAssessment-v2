from feature_learner import create_feature_learner
from imagetransform import ImageTransform
from dataset import create_ava_dataloader
from trainer import Trainer
from metrics import EMD, NLLLoss, CrossEntropyLoss
from utils import *


model_name = "eff3"
img_dir = r"D:\VXB\AVA\images"
unfreeze_after = 2
n_groups = 3
batch_size = 75
epochs = 50

device = default_device()

checkpoint_dir = rf"./results/checkpoints/classifier_" + model_name
state_file = join_path(checkpoint_dir, "summary.csv")

estimator, tf_info = create_feature_learner(model_name, n_output=3)
image_tf = ImageTransform(**tf_info)
print(":: Transformer:", image_tf, "\n")

optimizer_info = {
    "name": "Adam",
    "param_groups": [
        {"params": "features", "lr": 5e-5},
        {"params": "classifier", "lr": 5e-4},
    ],
}
lr_updater_info = {
    "strategy": "onplateau",
    "mode": "min",
    "patience": 10,
    "factor": 0.95,
    "verbose": True,
}

train_dl_info = {
    "images_dir": img_dir,
    "annotation_file": rf"./data/raw/val_labels.csv",  # r'./data/train.csv',#
    "batch_size": batch_size,
    "shuffle": True,
    "image_transform": image_tf.TrainTransform,
    "label_transform": lambda x: LabelTransform(x, splits=[0.0, 4.5, 6.5, 10.0]),
}
train_dl = create_ava_dataloader(train_dl_info)
valid_dl_info = {
    "images_dir": img_dir,
    "annotation_file": rf"./data/raw/test_labels.csv",  # r'./data/val.csv',#
    "batch_size": batch_size,
    "shuffle": False,
    "image_transform": image_tf.TestTransform,
    "label_transform": lambda x: LabelTransform(x, splits=[0.0, 4.5, 6.5, 10.0]),
}
valid_dl = create_ava_dataloader(valid_dl_info)

trainer = Trainer(optimizer_info, CrossEntropyLoss, train_dl, lr_updater_info, None)
trainer.set_model(estimator, None, device=device)

print_current_time()
trainer.train(
    epochs,
    sample_weights=None,
    unfreeze_after=unfreeze_after,
    verbose=True,
    checkpoint_dir=checkpoint_dir,
    state_file=state_file,
    validation_dataloader=valid_dl,
    start_epoch=0,  # trained_epoch+1
)
print_current_time()
