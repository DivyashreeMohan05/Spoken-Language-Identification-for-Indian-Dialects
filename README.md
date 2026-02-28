# Spoken-Language-Identification-for-Indian-Dialects
A spoken language identification system that classifies short audio segments into one of 22 Indian languages

Fine-tuning **wav2vec2-xls-r-300m** to classify short audio segments across **22 official Indian languages** (Assamese, Bengali, Hindi, Tamil, Telugu, Urdu, and more) — as part of a university ML project.

## Problem
With **400 samples and 5 speakers per language**, the core challenge is preventing the model from learning speaker identity instead of language features.

## Approach
- Baseline: supervised fine-tuning of `wav2vec2-xls-r-300m` on 22-class classification
- Improving over baseline (31.96% accuracy) via hyperparameter tuning, learning rate scheduling, and audio augmentation (pitch shifting, speed perturbation) to reduce speaker bias
- Analyzing confusion patterns between acoustically similar languages using t-SNE visualizations

## Constraints
`Max 600M parameters` · `No external datasets` · `No multimodal LLMs`

---
*Built on Kaggle · Dataset: subset of a multilingual Indian speech corpus via HuggingFace*
