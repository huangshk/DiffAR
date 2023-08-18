# DiffAR: Adaptive Conditional Diffusion Model for Temporal-augmented Human Activity Recognition

By [Shuokang Huang](https://github.com/huangshk), [Po-Yu Chen](https://github.com/bryan107) and [Julie A. McCann](https://wp.doc.ic.ac.uk/aese/people/julie-a-mccann/).

Official Implementation of [DiffAR: Adaptive Conditional Diffusion Model for Temporal-augmented Human Activity Recognition](https://www.ijcai.org/proceedings/2023/0424.pdf) (IJCAI 2023).

Full Paper with Appendix: [DiffAR_Full_Paper](DiffAR_Full_Paper.pdf)

## Abstract
Human activity recognition (HAR) is a fundamental sensing and analysis technique that supports diverse applications, such as smart homes and healthcare. In device-free and non-intrusive HAR, WiFi channel state information (CSI) captures wireless signal variations caused by human interference without the need for video cameras or on-body sensors. However, current CSI-based HAR performance is hampered by incomplete CSI recordings due to fixed window sizes in CSI collection and human/machine errors that incur missing values in CSI. To address these issues, we propose DiffAR, a temporal-augmented HAR approach that improves HAR performance by augmenting CSI. DiffAR devises a novel Adaptive Conditional Diffusion Model (ACDM) to synthesize augmented CSI, which tackles the issue of fixed windows by forecasting and handles missing values with imputation. Compared to existing diffusion models, ACDM improves the synthesis quality by guiding progressive synthesis with step-specific conditions. DiffAR further exploits an ensemble classifier for activity recognition using both raw and augmented CSI. Extensive experiments on four public datasets show that DiffAR achieves the best synthesis quality of augmented CSI and outperforms state-of-theart CSI-based HAR methods in terms of recognition performance.


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
```
@inproceedings{ijcai2023_diffar,
    title     = {DiffAR: Adaptive Conditional Diffusion Model for Temporal-augmented Human Activity Recognition},
    author    = {Huang, Shuokang and Chen, Po-Yu and McCann, Julie},
    booktitle = {Proceedings of the Thirty-Second International Joint Conference on
                 Artificial Intelligence, {IJCAI-23}},
    publisher = {International Joint Conferences on Artificial Intelligence Organization},
    editor    = {Edith Elkind},
    pages     = {3812--3820},
    year      = {2023},
    month     = {8},
    note      = {Main Track},
    doi       = {10.24963/ijcai.2023/424},
    url       = {https://doi.org/10.24963/ijcai.2023/424},
}
```
