# Mask-RCNN inference code and metrics calculation code for MIDL 

This repository contains scripts for performing inference nuclei segmentation using Detectron2, followed by metric calculations including F1 scores based on GeoJSON format results. This code is part of the submission to the Medical Imaging with Deep Learning (MIDL) conference.

## Set Up Environment 
In the following example we utilize Python >= 3.8 and CUDA version 11.7 on a Linux (or Windows) machine. For other CUDA versions, modify the installation of torch, torchvision and torchaudio accordingly (please refer to https://pytorch.org/ for all distributions).

```
python -m venv midl
source midl/bin/activate
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
pip install opencv-python
```
## Repository Structure

Below are the main directories in the repository: 

- `data/`: contains the checkpoint used for mask-rcnn inference

## Usage
1. `mask_rcnn_inference.py`: Runs inference on only the test set.  

```
python mask_rcnn_inference.py /path/to/images /path/to/output /path/to/model_weights.pth
```

2. `MIDL_calculate_f1_score.py`: Processes the GeoJSON output from the first script to calculate precision, recall and $F1$ scores per class. In addition micro and macro $F1$ score are calculated. Also usable for inference on NN192 geojsons and hovernet geojsons. 

```
MIDL_calculate_f1_score.py /path/to/ground_truth_folder /path/to/prediction_folder
```