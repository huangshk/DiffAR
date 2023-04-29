"""
[file]          preset.py
[description]   preset hyper-paremeters to configure the training and evaluation of ensemble classifier
"""
#
##
preset = {
    #
    ##
    "path": {
        #
        "data_incomplete_x": "./ACDM/Result/data_incomplete_x.npy",         # path of incomplete CSI
        "data_augment_x": "./ACDM/Result/data_augment_x.npy",               # path of augmented CSI
        "data_y": "./Data/data_y.npy",                                      # path of labels
        "index": "./Data/index_train_val_test.json",                        # path of preset index to split dataset
        "save": "./EnsembleClassifier/Result/",                             # path to save models and results
    },
    #
    ##
    "classifier": {
        "var_dim_model": 128,
        "var_num_head": 8,
        "var_dim_forward": 128,
        "var_num_layer": 2,
    },
    #
    ##
    "train": {
        #
        "epochs": 200,
        "batch_size": 16,
        "lr": 1e-4,
        #
        "len_sequence": 1200,       # length to align incomplete and augmented CSI
    },
}