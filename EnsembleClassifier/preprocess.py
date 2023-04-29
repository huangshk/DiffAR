"""
[file]          preprocess.py
[description]   ensemble dataset to include both incomplete CSI and augmented CSI
"""
#
##
import torch
import numpy as np
#
from sklearn.preprocessing import LabelEncoder
#
from preset import *










## ------------------------------------------------------------------------------------------ ##
## ----------------------------------- Ensemble Dataset ------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
class EnsembleDataset(torch.utils.data.Dataset):
    """
    <description>
    : dataset to train the ensemble classifier
    """
    #
    ##
    def __init__(self, 
                 data_x,        # np.array, shape (samples, features, time)
                 data_aug_x,    # np.array, shape (samples, features, time)
                 data_y,        # np.array, shape (samples, )
                 var_max_len):  
        
        """
        <parameter>
        : data_x: array of samples (incomplete), shape (samples, features, time)
        : data_aug_x: array of samples (augmented), shape (samples, features, time)
        : data_y: array of labels corresponding to data_x, shape (samples, )
        : var_max_len: max length of of samples
        """
        #
        ##
        super(EnsembleDataset, self).__init__()
        #
        self.data_x = []
        self.data_aug_x = []
        #
        self.data_y = torch.from_numpy(data_y)
        #
        ##
        for data_sample_x, data_sample_aug_x in zip(data_x, data_aug_x):
            #
            ##
            data_sample_x = torch.tensor(data_sample_x.astype(np.float32))
            data_sample_aug_x = torch.tensor(data_sample_aug_x.astype(np.float32))
            #
            ##
            data_sample_x = self.scale(data_sample_x)
            data_sample_aug_x = self.scale(data_sample_aug_x)
            #
            ##
            data_sample_x = self.align(data_sample_x, var_max_len)
            data_sample_aug_x = self.align(data_sample_aug_x, var_max_len)
            #
            self.data_x.append(data_sample_x)
            self.data_aug_x.append(data_sample_aug_x)
        #
        ##
        self.data_x = torch.stack(self.data_x)
        self.data_aug_x = torch.stack(self.data_aug_x)
        #
        self.var_num_sample = self.data_x.shape[0]
    
    #
    ##
    def __len__(self):
        #
        ##
        return self.var_num_sample
    
    #
    ##
    def __getitem__(self, var_i):
        #
        ##
        data_i_x = self.data_x[var_i]
        data_i_aug_x = self.data_aug_x[var_i]
        data_i_y = self.data_y[var_i]
        #
        return data_i_x, data_i_aug_x, data_i_y

    #
    ##
    def align(self, 
              data_sample_x,
              var_max_len):
        """
        <parameters>
        : data_sample_x: torch.tensor, shape (..., time)
        """
        #
        ##
        if data_sample_x.shape[-1] > var_max_len:
            #
            data_sample_x = data_sample_x[..., -var_max_len:]
        #
        elif data_sample_x.shape[-1] < var_max_len:
            #
            var_len = var_max_len - data_sample_x.shape[-1]
            data_sample_x = torch.nn.functional.pad(data_sample_x, (var_len, 0))
        #
        return data_sample_x

    #
    ##
    def scale(self, 
              data_sample_x):
        """
        <parameters>
        : data_sample_x: torch.tensor, shape (..., time)
        """
        #
        ##
        data_x_mean = torch.mean(data_sample_x, dim = -1, keepdim = True)
        data_x_std = torch.std(data_sample_x, dim = -1, keepdim = True)
        data_sample_x = (data_sample_x - data_x_mean) / data_x_std
        #
        return data_sample_x










#
##
def preprocess(data_x, data_aug_x, data_y, var_index):
    """
    <description>
    : function to preprocess data arrays
    : to training set, validation set and test set according to splitting index
    """
    #
    ##
    data_train_x = data_x[var_index["train"]]
    data_train_aug_x = data_aug_x[var_index["train"]]
    data_train_y = data_y[var_index["train"]]
    #
    ##
    data_val_x = data_x[var_index["val"]]
    data_val_aug_x = data_aug_x[var_index["val"]]
    data_val_y = data_y[var_index["val"]]
    #
    ##
    data_test_x = data_x[var_index["test"]]
    data_test_aug_x = data_aug_x[var_index["test"]]
    data_test_y = data_y[var_index["test"]]
    #
    ##
    var_label_encoder = LabelEncoder()
    data_train_y = var_label_encoder.fit_transform(data_train_y)
    data_val_y = var_label_encoder.transform(data_val_y)
    data_test_y = var_label_encoder.transform(data_test_y)
    #
    ##
    data_train_set = EnsembleDataset(data_train_x, data_train_aug_x, data_train_y,
                                     preset["train"]["len_sequence"])
    #                                 
    data_val_set = EnsembleDataset(data_val_x, data_val_aug_x, data_val_y,
                                   preset["train"]["len_sequence"])
    #                               
    data_test_set = EnsembleDataset(data_test_x, data_test_aug_x, data_test_y,
                                    preset["train"]["len_sequence"])
    #
    return data_train_set, data_val_set, data_test_set, var_label_encoder

