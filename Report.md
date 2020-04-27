# Project Report

## Hyperparameters
All the parameters we used can be found in train_waymo.yaml and train_tdt4265.yaml

## How to reproduce our results
Clone the repository, set up the Python/Anaconda environment as done in assignment 4, and install the requirements:
```
pip install -r requirements.txt
```
Run `setup_waymo.py` and `update_tdt4265_dataset.py`.

Train on the Waymo dataset:
```
python train.py configs/train_waymo.yaml
```
Transfer the model:
```
python transfer_learn.py configs/train_waymo.yaml
```
Train on the TDT4265 dataset:
```
python train.py configs/train_tdt4265.yaml
```
Finally run the evaluation script.

## Model architecture
For the backbone we use a ResNet-34 (pretrained on ImageNet), with some modifications, for the first three output feature banks. For the last three feature banks we used additional layers of ResNet BasicBlocks, each of size 2 and with the first block downsampling.

| Layer                                                           | Output                         |
|-----------------------------------------------------------------|--------------------------------|
| Conv2d(3, 64) kernel 7x7, stride 2x2, padding 3x3               |                                |
| BatchNorm2d(64)                                                 |                                |
| ReLU                                                            |                                |
| ResNet-34 layer 1-2 (extra downsampling BasicBlock at the end)  | Feature bank 1 (128 x 38 x 38) |
| ResNet-34 layer 3                                               | Feature bank 2 (256 x 19 x 19) |
| ResNet-34 layer 4                                               | Feature bank 3 (512 x 10 x 10) |
| Additional layer of two BasicBlocks (first downsampling)        | Feature bank 4 (256 x 5 x 5)   |
| Additional layer of two BasicBlocks (first downsampling)        | Feature bank 5 (256 x 3 x 3)   |
| Additional layer of two BasicBlocks (first downsampling)        |                                |
| Conv2d(3, 64) kernel 3x3, stride 2x2, padding 1x1               |                                |
| BatchNorm2d(128)                                                |                                |
| ReLU                                                            | Feature bank 6 (128 x 1 x 1)   |
