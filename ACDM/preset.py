"""
[file]          preset.py
[description]   preset hyper-paremeters to configure the training and evaluation of ACDM
"""
#
##
preset = {
    #
    ##
    "path": {
        "data": "./Data/data_x.npy",                    # path raw CSI
        "index": "./Data/index_train_val_test.json",    # path of preset index to split dataset
        "save": "./ACDM/Result/",                       # path to save models and augmented data
        "model": "./ACDM/Result/weight_0.pt",           # path to load models for evaluation
    },
    #
    ##
    "spectrogram": {
        "n_fft": 256,
        "hop_length": 64,
        "normalized": True,
        "power": 1.0,
    },
    #
    ##
    "acdm": {
        "var_num_residual": 10,
        "var_dim_residual": 32,
        "var_dim_feature": 90,
        "var_dim_step": 128,
        "var_max_step": 100,
        "var_conv_size_list": [1, 3, 5],
        "var_conv_dilation_cycle": [1, 2, 4, 8, 16],
    },
    #
    ##
    "train": {
        "epochs": 100000,
        "batch_size": 16,
        "lr": 1e-4,
        #
        "min_beta": 1e-5,
        "max_beta": 1e-2,
        "max_step": 100,
        #
        "mode": "forecast",     # forecast / imputation / mix
        "ratio_fc": 0.2,        # lambda_fc
        "ratio_im": 0.2,        # lambda_im
    },
}
