# Convergence Analysis
This folder contains code that demonstrates the increasing neural predictivity among Large Models.
The code is partially borrowed from [the repo of "The Platonic Representation Hypothesis" paper](https://github.com/minyoungg/platonic-rep/).

### Features Generation

First, extract the embeddings from Large Vision and Language Models using the `extract_model_features.py` file. To run:
```bash
python extract_model_features.py --gpu 0 --batch_size 128 --modality [vision/language] --model_family [vit/clip/...] --pooling [mean/cls]
```

Notes:
- Available model families can be found in `utils.py` file.

### Neural predictivity

After extracting model features, flatten and store the fMRI data by running:
```bash
python extract_brain_features.py --subj 1 
```

Then measure the neural predictivity of Large Models:
```bash
python measure_alignment.py --subj 1 --roi general --modality vision --model_family vit --pooling mean
```