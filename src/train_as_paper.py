from feature_learner import create_feature_learner
from imagetransform import ImageTransform
from dataset import create_ava_dataloader
from trainer import Trainer
from metrics import EMD
from utils import *

model_name = "eff3"

unfreeze_after = 2
device = default_device()

estimator, tf_info = create_feature_learner(model_name)
image_tf = ImageTransform(**tf_info)
print(":: Transformer:", image_tf, "\n")
img_dir = r"D:\VXB\AVA\images"
batch_size = 75

train_dl_info = {
    "images_dir": img_dir,
    "annotation_file": rf"./data/raw/train_labels.csv",  # r'./data/train.csv',#
    "batch_size": batch_size,
    "shuffle": True,
    "image_transform": image_tf.TrainTransform,
}
train_dl = create_ava_dataloader(train_dl_info)

valid_dl_info = {
    "images_dir": img_dir,
    "annotation_file": rf"./data/raw/val_labels.csv",  # r'./data/val.csv',#
    "batch_size": batch_size,
    "shuffle": False,
    "image_transform": image_tf.TestTransform,
}
valid_dl = create_ava_dataloader(valid_dl_info)

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
    "patience": 3,
    "factor": 0.7,
    "verbose": True,
}

trainer = Trainer(optimizer_info, EMD, train_dl, lr_updater_info, None)
checkpoint_dir = rf"./results/checkpoints/retrain_paper_" + model_name
state_file = join_path(checkpoint_dir, "summary.csv")

epochs = 50
# trainer.set_model(estimator, None, device=device) ###### CHANGE HERE

trained_epoch = 15  ###### CHANGE HERE
trainer.set_model(
    estimator, join_path(checkpoint_dir, f"epoch-{trained_epoch}.tar"), device=device
)  ###### CHANGE HERE

print_current_time()
trainer.train(
    epochs,
    None,
    unfreeze_after,
    verbose=True,
    checkpoint_dir=checkpoint_dir,
    state_file=state_file,
    validation_dataloader=valid_dl,
    start_epoch=trained_epoch,  ###### CHANGE HERE
)
print_current_time()
