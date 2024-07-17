
This repository is the code corresponding to the paper:

**Multi-level Fusion Network based on Electromagnetic Characteristics and SAR Images for SAR Target Recognition**

## Dataset

MSTAR (Moving and Stationary Target Acquisition and Recognition) Database and FUSAR-ship Database

[Example img of MSTAR](sample_img_MSTAR.JPG)

[Example img of FUSAR](sample_img_FUSAR.png)


## Model

The proposed model will be introduced in detail when the paper is accepted.


## Training

For training, this implementation details are set in `experiments/config/*.json`

##### Quick Start Guide for Training

 python src/train.py --configfile 'experiments/config/SOC.json'

## Details about the specific environment of this repository

| OS | Windows 11 |
| CPU | Intel i7-12700k |
| GPU | RTX 3060Ti 8GB |
| Memory | 16GB |
| SSD | 500GB |


## Acknowledgment

This code package is based in part on the source code of the [AConvNet-pytorch] repository.

[AConvNet-pytorch](https://github.com/jangsoopark/AConvNet-pytorch)