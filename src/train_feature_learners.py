from feature_learner import create_feature_learner
from imagetransform import ImageTransform
from dataset import create_ava_dataloader
from trainer import Trainer
from metrics import EMD
from utils import *

model_name = "eff3"

unfreeze_after = 2
device = default_device()

# group_name = "group_0"
for group_name in ["group_0", "group_2"]:
    estimator, tf_info = create_feature_learner(model_name)
    image_tf = ImageTransform(**tf_info)
    print(":: Transformer:", image_tf, "\n")

    img_dir = r"D:\VXB\AVA\images"
    batch_size = 75

    train_dl_info = {
        "images_dir": img_dir,
        "annotation_file": rf"./data/train_{group_name}.csv",  # r'./data/train.csv',#
        "batch_size": batch_size,
        "shuffle": True,
        "image_transform": image_tf.TrainTransform,
    }
    train_dl = create_ava_dataloader(train_dl_info)

    valid_dl_info = {
        "images_dir": img_dir,
        "annotation_file": rf"./data/val_{group_name}.csv",  # r'./data/val.csv',#
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
    # optimizer_info      =   {
    #     'name': 'SGD',
    #     'param_groups': [
    #         {'params': 'features', 'lr': 5e-4},
    #         {'params': 'classifier', 'lr': 5e-3}
    #     ],
    #     'momentum': 0.9
    # }
    lr_updater_info = {
        "strategy": "onplateau",
        "mode": "min",
        "patience": 3,
        "factor": 0.7,
        "verbose": True,
    }
    # lr_updater_info = {
    #     "strategy": "exponential",
    #     "update_freq": 5,
    #     "gamma": 0.95,
    #     "verbose": True,
    # }
    trainer = Trainer(optimizer_info, EMD, train_dl, lr_updater_info, None)

    checkpoint_dir = rf"./results/checkpoints/{group_name}_" + model_name
    state_file = join_path(checkpoint_dir, "summary.csv")

    epochs = 50

    trainer.set_model(estimator, None, device=device)
    # trained_epoch = 7
    # trainer.set_model(estimator, join_path(checkpoint_dir, f'epoch-{trained_epoch}.tar'), device=device)

    print_current_time()
    trainer.train(
        epochs,
        None,
        unfreeze_after,
        verbose=True,
        checkpoint_dir=checkpoint_dir,
        state_file=state_file,
        validation_dataloader=valid_dl,
        start_epoch=0,  # trained_epoch+1
    )
    print_current_time()
