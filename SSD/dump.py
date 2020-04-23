from ssd.config.defaults import cfg
from ssd.data.build import make_data_loader
from ssd.modeling.detector import SSDDetector

# config
cfg.MODEL.BACKBONE.NAME = 'resnet34'
cfg.INPUT.IMAGE_SIZE = 300
cfg.MODEL.BACKBONE.OUT_CHANNELS = (256,512,512,256,256,128) # resnet34
# cfg.MODEL.BACKBONE.OUT_CHANNELS = (256,512,256,256,128,64) # wip34
cfg.MODEL.PRIORS.FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
cfg.SOLVER.MAX_ITER = 120000
cfg.SOLVER.LR_STEPS = [80000, 100000]
cfg.SOLVER.BATCH_SIZE = 1
cfg.SOLVER.LR = 1e-3
cfg.OUTPUT_DIR = "outputs/dump"
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



