from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
import cv2


# 读取测试图像
im = cv2.imread("./test.jpg")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))  # 使用之前训练的配置文件
cfg.MODEL.WEIGHTS = "./model_final.pth"  # 使用之前训练得到的模型权重文件

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

print(outputs["instances"].pred_boxes)

# 创建自定义的 MetadataCatalog
my_metadata = MetadataCatalog.get("my_dataset")
my_metadata.set(thing_classes=["plant"])  # 设置自定义的类别名称列表

# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], my_metadata, scale=1.0)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
out_frame=out.get_image()[:, :, ::-1]
cv2.imwrite("./result.png",out_frame)
