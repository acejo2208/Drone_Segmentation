# IsLab Drone Projects
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class DroneSegDataset(CocoDataset):

    CLASSES = ('lawn, flower_garden', 'forest', 'liver', 'road', 'pavement', 
               'parking_lot', 'crosswalk', 'hiking_trail', 'trail', 'flower_bed')