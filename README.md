# Decoding Neural Activity into Meaningful Captions via Large Vision-Language Models

This repository contains code for decoding brain activity patterns into natural language descriptions using Large Vision-Language Models, specifically BLIP-2. 

The project demonstrates how modern large vision and language models predict the neural responses better and how vision-language models can serve as effective decoders for neural activity, enabling translation of brain signals into natural language.

## Setup & Installation

> **Note**: Please check and update directory paths according to your environment.

Create a Python 3.10 environment and install dependencies via:

```bash
pip install -r requirements.txt
```

## Project Structure

### 1. Data Preparation

The project uses the Natural Scenes Dataset (NSD) and COCO dataset. Follow the instructions in  [data_scripts](./data_scripts/) to:

- Download the NSD and COCO datasets
- Process and prepare the data for model training
- Set up the required directory structure

It's **must-do** step. See [data_scripts readme.md](./data_scripts/readme.md) for details.

### 2. Convergence Analysis

Our research demonstrates that as AI models become more sophisticated, their ability to predict neural responses improves. The [convergence_analysis](./convergence_analysis/) folder contains code to:
- Extract features from various Large Models
- Analyze neural predictivity using linear regression
- Compare performance across different model architectures
- Visualize the alignment between model representations and brain activity

This analysis demonstrates the growing alignment between neural responses and representations learned by modern AI architectures.

See [readme file](./convergence_analysis/readme.md) in the `convergence_analysis` folder for more details.

### 3. Brain Decoding

The [neuro-language-models](./neuro-language-models/) folder contains the core implementation for translating fMRI signals into natural language descriptions. Key features include:

- Feature extraction using BLIP-2
- Two-stage training pipeline:
    1. Alignment training with Ridge Regression
    2. Contrastive learning for improved semantic mapping
- Inference pipeline for generating captions from brain activity
- Generated captions in CSV files and retrieval results

See [neuro-language-models readme](./neuro-language-models/readme.md) for more details.

## Citation

TBD.