from feature_learner import create_feature_learner
from imagetransform import ImageTransform
from dataset import create_ava_dataloader
from tester import Tester
from utils import default_device, join_path

model_name = "eff3"
device = default_device()
img_dir = r"D:\VXB\AVA\images"
batch_size = 75

# epochs = {"group_0": 10, "group_1": 8, "group_2": 12}
epochs = 7

checkpoint_file = join_path(
    rf"./results/checkpoints/retrain_paper_" + model_name, f"epoch-{epochs}.tar"
)

estimator, tf_info = create_feature_learner(model_name)
image_tf = ImageTransform(**tf_info)
# print(":: Transformer:", image_tf, "\n")
tester = Tester(estimator, checkpoint_file)

prediction_file = join_path(
    rf"./results/predictions/retrain_paper_" + model_name, f"epoch_{epochs}.csv"
)
test_dl_info = {
    "images_dir": img_dir,
    "annotation_file": rf"./data/raw/test_labels.csv",  # r"./data/test.csv",  #
    "batch_size": batch_size,
    "shuffle": False,
    "image_transform": image_tf.TestTransform,
}
test_dl = create_ava_dataloader(test_dl_info)
tester.predict_save(test_dl, prediction_file, save_label=True, device=device)
