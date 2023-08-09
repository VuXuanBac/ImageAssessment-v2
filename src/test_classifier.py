from feature_learner import create_feature_learner
from imagetransform import ImageTransform
from dataset import create_ava_dataloader
from tester import Tester
from utils import default_device, join_path, LabelTransform

model_name = "eff3"
device = default_device()
img_dir = r"D:\VXB\AVA\images"
batch_size = 75

epochs = {
    "Extract/t2_classifier_v2": 4,
    "Extract/t2_classifier_v3": 7,
}

estimator, tf_info = create_feature_learner(model_name, n_output=3)
image_tf = ImageTransform(**tf_info)

test_dl_info = {
    "images_dir": img_dir,
    "annotation_file": rf"./data/raw/test_labels.csv",  # r'./data/val.csv',#
    "batch_size": batch_size,
    "shuffle": False,
    "image_transform": image_tf.TestTransform,
    "label_transform": lambda x: LabelTransform(x, splits=[0.0, 4.5, 6.5, 10.0]),
}
test_dl = create_ava_dataloader(test_dl_info)

for k, v in epochs.items():
    checkpoint_file = join_path("./results/checkpoints/", k, f"epoch-{v}.tar")

    # print(":: Transformer:", image_tf, "\n")
    tester = Tester(estimator, checkpoint_file)

    prediction_file = join_path("./results/predictions/", k + f"-epoch_{v}.csv")

    tester.predict_save(test_dl, prediction_file, save_label=True, device=device)
