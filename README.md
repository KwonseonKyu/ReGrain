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

- **Input image**
  
<img src="https://github.com/user-attachments/assets/93be0ef5-f15c-47f3-aced-d0cd8cce456e" alt="org" width="700"/>


- **Target image (Grain intensity - 0.1)**
  
<img src="https://github.com/user-attachments/assets/180a2229-e4ef-4108-b2f6-d4bb57ccbfb1" alt="fg" width="700"/>


### OpenCV & ffmpeg Grain image

- **Film-grain with OpenCV (Grain intensity - 0.1)**
  
<img src="https://github.com/user-attachments/assets/f926a068-164a-48a0-bd02-27e3e6686cdd" alt="opencv" width="700"/>


- **Film-grain with ffmpeg (Noise intensity - 20)**
  
<img src="https://github.com/user-attachments/assets/1cdd7540-67ba-47bb-bd92-2af5d94f06a3" alt="ffmpeg" width="700"/>


### DeepFilmGrain pretrained model

- **Film-grain with DFG model (Grain intensity - 0.1)**
  
<img src="https://github.com/user-attachments/assets/292c385f-790c-477c-be83-08acb985237c" alt="DFG" width="700"/>


## My re-grain model test

### Image 1

#### Original Image
<img src="https://github.com/user-attachments/assets/6a24793f-1519-4258-a3c8-93620d350a2e" alt="Original BSD_2092" width="300"/>

#### Grain Images

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/7d6bbbcd-5b43-477d-ba51-2c1d646fd926" alt="Grain Image 25%" width="300"/>
      <br>25% Grain Level
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/4c89dcff-649d-44d0-add3-c30def51c188" alt="Grain Image 50%" width="300"/>
      <br>50% Grain Level
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/b5d1a532-b13a-4a64-a510-0824a759cc4d" alt="Grain Image 75%" width="300"/>
      <br>75% Grain Level
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/177cacb1-91e2-41d7-bac1-74017441b305" alt="Grain Image 100%" width="300"/>
      <br>100% Grain Level
    </td>
  </tr>
</table>



### Image 2






