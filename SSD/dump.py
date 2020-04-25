from ssd.config.defaults import cfg
from ssd.data.build import make_data_loader
from ssd.modeling.detector import SSDDetector

# config
cfg.MODEL.BACKBONE.NAME = 'resnet34'
# cfg.INPUT.IMAGE_SIZE = 300
cfg.INPUT.IMAGE_SIZE = (480,640)
cfg.MODEL.BACKBONE.OUT_CHANNELS = (128,256,512,256,256,128) # resnet34
# cfg.MODEL.PRIORS.FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
cfg.MODEL.PRIORS.FEATURE_MAPS = [(30,40), (15,20), (8,10), (4,5), (2,3), (1,1)]
cfg.SOLVER.BATCH_SIZE = 2
cfg.DATASET_DIR = "datasets"
cfg.DATASETS.TRAIN = ("waymo_train",)
# cfg.DATASETS.TEST = ("waymo_val",)

model = SSDDetector(cfg)
for level, bank in enumerate(model.backbone.feature_extractor):
    bank_n = level+1
    print("Bank %d:" % bank_n, bank)

data_loader = make_data_loader(cfg, is_train=True, max_iter=cfg.SOLVER.MAX_ITER)

images, targets, _ = next(iter(data_loader)) # 1 batch
outputs = model(images, targets=targets)



