from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
import cv2
import time

im = cv2.imread("test.png")

cfg = get_cfg()
cfg.merge_from_file("./detectron2_dc/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
cfg.MODEL.WEIGHTS = "./output/bean1_mask_rcnn_R_50_FPN_3x_500iter/model_final.pth"
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
predictor = DefaultPredictor(cfg)
t1=time.time()
outputs = predictor(im)
t2=time.time()
print(t2-t1)


print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

# 创建自定义的 MetadataCatalog
my_metadata = MetadataCatalog.get("my_dataset")
my_metadata.set(thing_classes=["leaf"])  # 设置自定义的类别名称列表

v = Visualizer(im[:, :, ::-1], my_metadata, scale=1.0)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
out_frame=out.get_image()[:, :, ::-1]

cv2.imshow(out_frame)