from feature_learner import create_feature_learner
from imagetransform import ImageTransform
from dataset import create_ava_dataloader
from tester import Tester
from utils import default_device

model_name = "eff3"
device = default_device()
img_dir = r"D:\VXB\AVA\images"
batch_size = 75

# epochs = {"group_0": 10, "group_1": 8, "group_2": 12}
epochs = {0: 10, 1: 7, 2: 12}
for group in [2]:
    checkpoint_file = rf"./results/checkpoints/group_{group}_eff3/epoch-{epochs[group]}.tar"

    estimator, tf_info = create_feature_learner(model_name)
    image_tf = ImageTransform(**tf_info)
    # print(":: Transformer:", image_tf, "\n")
    tester = Tester(estimator, checkpoint_file)

    for dset in [f"{phase}_group_{i}.csv" for phase in ["test", "val"] for i in [2,0,1]]:
        prediction_file = rf"./results/predictions/group_{group}-epoch_{epochs[group]}-{dset}"

        test_dl_info = {
            "images_dir": img_dir,
            "annotation_file": rf"./data/{dset}", # r"./data/test.csv",  # 
            "batch_size": batch_size,
            "shuffle": False,
            "image_transform": image_tf.TestTransform,
        }
        test_dl = create_ava_dataloader(test_dl_info)

        tester.predict_save(test_dl, prediction_file, save_label=True, device=device)
