# DiffAR: Adaptive Conditional Diffusion Model for Temporal-augmented Human Activity Recognition

By [Shuokang Huang](https://github.com/huangshk), [Po-Yu Chen](https://github.com/bryan107) and [Julie A. McCann](https://wp.doc.ic.ac.uk/aese/people/julie-a-mccann/).

Official PyTorch implementation of our IJCAI 2023 paper "DiffAR: Adaptive Conditional Diffusion Model for Temporal-augmented Human Activity Recognition".

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

- Four public datasets are adopted in this paper to evaluate the performance of DiffAR.
    - [Office](https://github.com/ermongroup/Wifi_Activity_Recognition)
    - [SignFi](https://github.com/yongsen/SignFi)
    - [Interactions](https://data.mendeley.com/datasets/3dhn4xnjxw/1)
    - [Widar 3.0](https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset)

- Please put the datasets in the `Data` directory (which you need to create in advance) and preprocess them as mentioned in our paper.

- For convenience, we provide the preprocessed Office dataset [here](https://drive.google.com/drive/folders/17pNQvkFx0juZMPf0P-Su0uyS-f7cfua1). You can download it to the `Data` directory for a quick start.

### ACDM: Training and Evaluation

- To tune the hyperparameters of ACDM, please modify `ACDM/preset.py` before training and evaluation.

- To train ACDM in a self-supervised manner, please run `python ACDM/train.py`. The weights of trained ACDM will be stored in `ACDM/Result/`. You need to create this directory in advance.

- To evaluate ACDM, please run `python ACDM/evaluate.py`. The output will indicate the evaluation results, in terms of Mean Absolute Error (MAE), Mean Squared Error (MSE) and Continuous Ranked Probability Score (CRPS).

- To apply ACDM for CSI augmentation, please run `python ACDM/augment.py`. The augmented CSI will be stored in `ACDM/Result/`.

### Ensemble Classifier: Training and Evaluation

- To tune the hyperparameters of Ensemble Classifier, please modify `EnsembleClassifier/preset.py` before training and evaluation.

- To evaluate Ensemble Classifier, please run `python EnsembleClassifier/evaluate.py`. The results will be presented in the output and written to `result_test.csv`.

## Citation
If you find this code useful for your research, please consider citing the following paper:

