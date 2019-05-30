import os
import tensorflow as tf
import mrcnn.model as modellib
import warnings
from EwattDataset import EwattDataset
from EwattConfig import EwattConfig

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

# 获取根目录
ROOT_DIR = os.getcwd()
# print(ROOT_DIR)
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

dataset_train = EwattDataset()
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

dataset_path = os.path.join(ROOT_DIR,"data")
config = EwattConfig()
def train():
    dataset_train = EwattDataset()
    dataset_train.load_antenna(dataset_path, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = EwattDataset()
    dataset_val.load_antenna(dataset_path, "val")
    dataset_val.prepare()

    model = modellib.MaskRCNN(mode='training',config=config,model_dir=DEFAULT_LOGS_DIR)
    model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='all')
train()