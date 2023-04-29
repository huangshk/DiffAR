# DiffAR: Adaptive Conditional Diffusion Model for Temporal-augmented Human Activity Recognition

By [Shuokang Huang](https://github.com/huangshk), [Po-Yu Chen](https://github.com/bryan107) and [Julie A. McCann](https://wp.doc.ic.ac.uk/aese/people/julie-a-mccann/).

Official implementation of our IJCAI 2023 paper "DiffAR: Adaptive Conditional Diffusion Model for Temporal-augmented Human Activity Recognition"

## Introduction

<div align = "center">
    <a href="./">
        <img src="./Figure/DiffAR.svg" width = "100%"/>
    </a>
</div>

## Environment

- Python = 3.9.13
- PyTorch = 1.13.0
- NumPy = 1.21.5

To install all the required packages, please run `pip install -r requirements.txt`.

## Usage

### Dataset

### ACDM: Training and Evaluation

```
python ACDM/train.py
```

```
python ACDM/evaluate.py
```

```
python ACDM/augment.py
```

### Ensemble Classifier: Training and Evaluation

```
python EnsembleClassifier/evaluate.py
```

## Citation
