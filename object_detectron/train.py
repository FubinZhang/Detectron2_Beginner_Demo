import os
import numpy as np
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
import torch

#set up cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

register_coco_instances("plant_train", {}, "./dataset/train/dataset.json", "./dataset/train")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")) #配置预训练模型
cfg.DATASETS.TRAIN = ("plant_train",) # 配置训练集
cfg.DATASETS.TEST = () # 配置测试集
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
cfg.DATALOADER.NUM_WORKERS = 2 # 线程数
cfg.SOLVER.IMS_PER_BATCH = 2 # 每次迭代的图像数量
cfg.SOLVER.BASE_LR = 0.00025 # 初始学习率
cfg.SOLVER.MAX_ITER = 1 # 训练迭代次数
cfg.SOLVER.STEPS = [] # 学习率的下降策略
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # 每张训练的ROI数量
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 根据实际类别数量修改
cfg.OUTPUT_DIR = './output/plant_object_fpn_R_50_1x_1' # 模型输出保存路径
cfg.MODEL.DEVICE = "cuda"



# 训练器定义
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False) # 是否继承上一次训练的参数，在自己的output里面
trainer.train()
