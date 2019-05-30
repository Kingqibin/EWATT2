from mrcnn.config import Config
class EwattConfig(Config):
    NAME = 'ewatt'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (16 * 16, 32 * 128, 64 * 128, 32 * 256, 64 * 512)
    TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 20
    DETECTION_MIN_CONFIDENCE = 0.9


