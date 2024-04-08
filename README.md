# Mask-RCNN inference code and metrics calculation code for MIDL 

This repository contains scripts for performing nuclei segmentation using Detectron2, followed by metric calculations including F1 scores based on GeoJSON format results. This code is part of the submission to the Medical Imaging with Deep Learning (MIDL) conference.

## Set Up Environment

```
conda env create -f environment.yml
conda activate midl
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

```
## Repository Structure

Below are the main directories in the repository: 

- `data/`: the checkpoint used for mask-rcnn inference

## Usage
1. `mask_rcnn_inference.py`: Runs inference on only the test set.  

```
run python mask_rcnn_inference.py /path/to/images /path/to/output /path/to/model_weights.pth
```

2. `MIDL_calculate_f1_score.py`: Processes the GeoJSON output from the first script to calculate precision, recall and $F1$ scores per class. In addition micro and macro $F1$ score are calculated. Also usable for inference on NN192 geojsons and hovernet geojsons. 

```
python process_geojson.py /path/to/ground_truth_folder /path/to/prediction_folder
```