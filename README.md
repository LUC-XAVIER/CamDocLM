# CamDocLM

Fine-tuning LayoutLM on synthetic Cameroon NICs and passports generated with SynthText.

## Overview
This project combines [SynthText](https://github.com/ankush-me/SynthText) for synthetic dataset generation with [LayoutLM](https://huggingface.co/microsoft/layoutlm-base-uncased) for document understanding.  
The goal is to build a model that can process and classify structured identity documents (NICs, passports).

## Environment Setup
Create a Conda environment with Python 3.11:
```bash
conda create -n camdoclm_env python=3.11 -y
conda activate camdoclm_env
pip install torch torchvision torchaudio
pip install transformers datasets huggingface_hub
pip install opencv-python pillow numpy matplotlib scipy shapely tqdm
```

## Project Structure:
```
CamDocLM/
│── data/              # Generated NICs & passports
│── configs/           # SynthText + training configs
│── scripts/           # Preprocessing & training scripts
│── notebooks/         # Experiment notebooks
│── external/
│    └── SynthText/    # Cloned SynthText repo (ignored in git)
│── README.md
│── .gitignore
```

## Workflow
Clone this repo:
```
git clone https://github.com/yourusername/CamDocLM.git
cd CamDocLM
```
Clone SynthText inside SynthText_Service (not tracked by git):

```
mkdir SynthText_Service
cd SynthText_Service
git clone https://github.com/ankush-me/SynthText.git
cd ..
```
