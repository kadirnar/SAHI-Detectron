from detectron2.config import get_cfg
from sahi.model import Detectron2DetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import download_from_url
from sahi.utils.detectron2 import export_cfg_as_yaml

# Download the model and images
model_path = download_from_url(
    "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl",
    "model_final_280758.pkl")

image = download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg',
                  'demo_data/small-vehicles1.jpeg')

cfg = get_cfg()
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 445
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2

config_path = export_cfg_as_yaml(cfg, export_path='config.yaml')

# test model
detection_model = Detectron2DetectionModel(
    model_path=model_path,
    config_path=config_path,
    confidence_threshold=0.2,
    image_size=640,
    device="cpu",  # or 'cuda:0'
)

result = get_sliced_prediction(
    image,  # image _path
    detection_model,
    slice_height=320,
    slice_width=320,
    overlap_height_ratio=0.4,
    overlap_width_ratio=0.4,
)
# Sahi+ Detectron2
result.export_visuals(export_dir="demo_data/")
