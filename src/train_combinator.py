from combinator import Combinator
from dataset import create_merge_dataloader
from trainer import Trainer
from metrics import EMD
from utils import *

model_name = "eff3"
n_groups = 3
batch_size = 128
epochs = 100

device = default_device()


#### preprocessing
def concat_group_prediction():
    for phase in ["val", "test"]:
        for group in range(3):
            concat_save(
                [
                    rf"./results/predictions/group_{group}--{phase}_group_{dgr}.csv"
                    for dgr in range(3)
                ],
                rf"./results/predictions/group_{group}--merge_{phase}.csv",
                dtypes={0: int},
                column_cut=slice(None, 10),
            )
        concat_save(
            [rf"./data/{phase}_group_{gr}.csv" for gr in range(3)],
            rf"./data/merge_{phase}.csv",
        )


# concat_group_prediction()

#######
checkpoint_dir = rf"./results/checkpoints/combinator_" + model_name
state_file = join_path(checkpoint_dir, "summary.csv")

train_inputs = [
    rf"./results/predictions/group_{group}--merge_val.csv" for group in range(3)
]
train_dl = create_merge_dataloader(
    train_inputs,
    batch_size,
    merge_on=0,
    label_anno_file=rf"./data/merge_val.csv",
    shuffle=True,
)

valid_inputs = [
    rf"./results/predictions/group_{group}--merge_test.csv" for group in range(3)
]
valid_dl = create_merge_dataloader(
    valid_inputs,
    batch_size,
    merge_on=0,
    label_anno_file=rf"./data/merge_test.csv",
    shuffle=False,
)

estimator = Combinator([10 * n_groups, 300, 300, 10])

optimizer_info = {
    "name": "Adam",
    "param_groups": [{"params": "", "lr": 1e-3}],
}
lr_updater_info = {
    "strategy": "onplateau",
    "mode": "min",
    "patience": 10,
    "factor": 0.95,
    "verbose": True,
}

trainer = Trainer(optimizer_info, EMD, train_dl, lr_updater_info, None)
trainer.set_model(estimator, None, device=device)

print_current_time()
trainer.train(
    epochs,
    sample_weights=None,
    verbose=True,
    checkpoint_dir=checkpoint_dir,
    state_file=state_file,
    validation_dataloader=valid_dl,
    start_epoch=0,  # trained_epoch+1
)
print_current_time()
