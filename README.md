# FaciesSAM: A Segment Anything Model for Seismic Facies Segmentation

## Architecture
<img src='figures/arc.png'>

## Dataset
- All data used for training, validating and testing can be downloaded [here](https://drive.google.com/drive/folders/1uwUPxaNpUfTBIfLldxgFNcdTaoJ8zR7D?usp=sharing). 

- Data directory should be: 
```python3
  data   
    ├── sa.yaml
    ├── images
    │    ├── train
    │    │    ├── image_inline_300.jpg
    │    │    └── ...
    │    ├── val
    │    │    ├── test1_image_inline_100.jpg
    │    │    └── ...
    │    └── test
    │         ├── test2_image_inline_100.jpg
    │         └── ...
    └── labels
    |    ├── train
    |    │    ├── image_inline_300.txt
    |    │    └── ...
    |    ├── val
    |    │    ├── test1_image_inline_100.txt
    |    │    └── ...
    |    └── test
    |    |    ├── test2_image_inline_100.txt
    |    |    └── ...
    |    └── test.cache
    |    |
    |    └── train.cache
    |    |
    |    └── val.cache   
    └── masks
        ├── train
        │    ├── image_inline_300.png
        │    └── ...
        ├── val
        │    ├── test1_image_inline_100.png
        │    └── ...
        └── test
            ├── test2_image_inline_100.png
            └── ...
```
## Requirements
Install requirements
```shell
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```
+ +NVIDIA GPU + CUDA CuDNN
+ +Linux (Ubuntu)
+ +Python v3.10
## Get the validation/test instance segmentation results
Open a terminal and run
```shell
python test_instance.py\
        --model_path 'models/FaciesSAM-x.pt'\ # model trained on image size of 640
        --cfg 'data/sa.yaml'\
        --split 'val'\
        --img_sz 640 # Image size can either be 320 or 640
```
This automatically create a new directory called `run`. Navigate to see results
<img src='figures/instance.png'>

##  Get the validation/test semantic segmentation results
Open terminal and run
```shell
python test_semantic.py\
        --model_path 'models/FaciesSAM-x.pt'\
        --data_path 'data/sa.yaml'\
        --split 'test'\
        --img_sz 640  # Image size can either be 320 or 640
```
This automatically create a new directory called `run`. Navigate to see results
<img src='figures/semantic.png'>

## How to train FaciesSAM on your dataset
Open terminal and run
```shell
python train.py \
        --model_path 'models/FaciesSAM-x.pt' \
        --cfg 'data/sa.yaml' \
        --aug True \
        --num_freeze 0 \
        --epochs 100 \
        --bs 16 \
        --img_sz 640 # Image size can either be 320 or 640

```
This automatically create a new directory called `run`. Navigate to see results

## Interact with seismic image through prompt action
Navigate to the `app` directory to see how FaciesSAM+CLIP can allow you interact with seismic images 

## Citation
```text
@article{atolagbe2024faciessam,
  title={FaciesSAM: A Segment Anything Model for Seismic Facies Segmentation},
  author={Joshua Atolagbe and Ardiansyah Koeshidayatullah},
  journal=computer&geosciences,
  year={2024}
}
```
## Credit
The codes in this repository are based on [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM/tree/main) and [Ultralytics](https://github.com/ultralytics/ultralytics)