# import required functions, classes
from detectron2.config import get_cfg
from sahi.model import Detectron2DetectionModel
from sahi.predict import get_sliced_prediction

# create config file

model_path = "model.pkl"
cfg = get_cfg()
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 445
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2

from sahi.utils.detectron2 import export_cfg_as_yaml
config_path = export_cfg_as_yaml(cfg, export_path='config.yaml')

# test model
detection_model = Detectron2DetectionModel(
    model_path=model_path,
    config_path=config_path,
    confidence_threshold=0.2,
    image_size=640,
    device="cuda:0",  # or 'cuda:0'
)

result = get_sliced_prediction(
    "demo_data/small-vehicles1.jpeg",  # image _path
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.4,
    overlap_width_ratio=0.4,
)
# Sahi+ Detectron2
result.export_visuals(export_dir="demo_data/")
