# Towards an Efficient Remote Sensing Image Compression Network with Visual State Space Model

## Architectures
The overall framework.

<img src="./assets/framework.png"  style="zoom: 33%;" />

The proposed the Omni-Selective Scan Module.

<img src="./assets/ossm.png"  style="zoom: 33%;" />

## Reconstructed Samples
<img src="./assets/visual.png"  style="zoom: 33%;" />

## Usage

### Train
We use the ImageNet, and COCO, the Aerial Image Dataset (AID) and the Northwestern Polytechnical University Very High Resolution 10-Class Dataset (NWPU VHR-10) for training.

Run the script for a simple training [MSE] pipeline:
```bash
python train.py -m vmic -d path/of/training/dataset] --epochs 400 -lr 1e-4 -q 6 --lambda 0.048 --batch-size 8 --cuda --gpu-id 0  --save --save_path path/to/save/model
```

### Testing
We selected AID and NWPU VHR-10 datasets, as well as panchromatic and multispectral images from the WorldView-3 dataset to Test.

Run the script for a simple testing pipeline:
``` 
python -m eval_model checkpoint [path of the pretrained checkpoint] -a vmic -p [path of testing dataset] --cuda 
```
