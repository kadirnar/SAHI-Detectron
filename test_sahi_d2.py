# import required functions, classes
from detectron2.config import get_cfg
from sahi.model import Detectron2DetectionModel
from sahi.predict import get_sliced_prediction, get_prediction
from sahi.utils.cv import read_image

# create config file

model_path = "mask_rcnn_R_50_C4_3x.pth"
cfg = get_cfg()
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 445
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

from sahi.utils.detectron2 import export_cfg_as_yaml

export_cfg_as_yaml(cfg, export_path='config.yaml')

# Test 
config_path = "config.yaml"

detection_model = Detectron2DetectionModel(
    model_path=model_path,
    config_path=config_path,
    confidence_threshold=0.5,
    image_size=640,
    device="cpu",  # or 'cuda:0'
)
# Only Model Test
result = get_prediction("demo_data/image.jpg", detection_model)  # image_path
result = get_prediction(read_image("demo_data/image.jpg"), detection_model)  # image_path
result.export_visuals(export_dir="demo_data/")

result = get_sliced_prediction(
    "image.jpg",  # image _path
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
# Sahi+ Detectron2
result.export_visuals(export_dir="demo_data/")
