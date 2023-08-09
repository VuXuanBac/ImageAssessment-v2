from combinator import Combinator
from dataset import create_merge_dataloader
from tester import Tester
from utils import default_device, join_path

model_name = "eff3"
device = default_device()
img_dir = r"D:\VXB\AVA\images"
batch_size = 512
n_groups = 3

test_inputs = [
    rf"./results/predictions/group_{group}--merge_test.csv" for group in range(3)
]
test_dl = create_merge_dataloader(
    test_inputs,
    batch_size,
    merge_on=0,
    label_anno_file=rf"./data/merge_test.csv",
    shuffle=False,
)


# epochs = {"group_0": 10, "group_1": 8, "group_2": 12}
epochs = {
    # "Extract/t2_combinator_v7": 10,
    "Extract/t2_combinator_v9": 78,
    "Extract/t2_combinator_v7": 94,
}

estimator = Combinator([10 * n_groups, 300, 300, 10])


for k, v in epochs.items():
    checkpoint_file = join_path("./results/checkpoints/", k, f"epoch-{v}.tar")

    # print(":: Transformer:", image_tf, "\n")
    tester = Tester(estimator, checkpoint_file)

    prediction_file = join_path("./results/predictions/", k + f"-epoch_{v}.csv")

    tester.predict_save(test_dl, prediction_file, save_label=True, device=device)
