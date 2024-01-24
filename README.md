# Detectron2_Beginner_Demo(detectron2入门样例)

一份关于detectron2入门级（训练+预测）的代码demo（目标检测/实例分割/全景分割........）

## 1.install

安装视觉库

`pip install opencv-python`

安装torch+torchvision，torch官网有可选配置的安装指令提供，以及Previous versions提供

`https://pytorch.org/`

安装detectron2，这里建议直接源码编译，以及要注意看清楚官方安装文档中的环境版本要求

`https://detectron2.readthedocs.io/en/latest/tutorials/install.html`

```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

## 2.数据集标注

工具：labelme

`pip install labelme`

安装后在环境终端输入指令labelme即可使用

### （1）object_detection

在labelme中create rectangle即可,

每一张image都会生成一个 `.json`文件，

转换格式可通过很多其他开源项目转化，不过我记得有一个项目直接可以 `pip install labelme2coco`使用（适用于目标检测和实例分割）

`https://github.com/hddlovefxx/labelme2coco`

最后会把所有的 `.json`都合并成一个文件

### （2）instance_segmentation

在labelme中create polygon即可

转化方法同目标检测相同

### （3）pannramic_segmentation

待写

## 3.训练（train.py）
