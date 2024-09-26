# ReGrain

## Overview
The film grain effect reproduces the properties of analog film, adding an aesthetic element to the image. The model produces a simple yet accurate grain image.

There are five intensitys of grain (0.01, 0.025, 0.05, 0.075, 0.1).

## Requirements
- CUDA 11.0
- cuDNN 8.0.5
- Python 3.8.19


## Installation

Download repository:
```bash
git clone https://github.com/KwonseonKyu/ReGrain.git
cd DeepFilmGrain_pytorch/
```

```bash
conda env create --name regrain --file environment.yml
conda activate regrain
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge anaconda cudnn==8.0.5
```

## Structure of the repository
- `environment.yml`: conda environment specifications and required packages 
- `src`: 
	- `data.py`: data loader
	- `model.py`: models builder
	- `utils.py`: utility functions
	- `losses.py`: loss functions 
	- `train.py`: training loops.
	- `test.py`: script for evaluating a trained model on a test dataset
	- `test_one.py`: script for evaluating a trained model on a single test image

File config.json contains the following training parameters:


## Download 

Download pretrained 


## File Paths

We also provide a FilmGrain dataset that is available at the address:


The structure of the dataset is the following:

```
    FilmGrainDataset
    ├── org				
    └── fg             
        ├── 01     
        ├── 025              
        ├── 05         
        ├── 075          
        └── 1     
```

## Train 

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train.py  
```

## Test 
```bash
python src/test.py  
```

## Test example (Berkeley segmentation dataset) 

### Input image & Target image
![org](https://github.com/user-attachments/assets/772d0c4f-b130-40c3-bf06-93c833fb94d9)



### OpenCV & ffmpeg Grain image



### DeepFilmGrain pretrained model



### ReGrain simple pretrained model



