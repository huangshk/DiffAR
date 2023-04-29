"""
[file]          preprocess.py
[description]   establish diffusion dataset; provide functions to mask/pad raw samples
"""
#
##
import torch
import numpy as np
from preset import preset










## ------------------------------------------------------------------------------------------ ##
## ---------------------------------- Diffusion Dataset ------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
class DiffusionDataset(torch.utils.data.Dataset):
    """
    <description>
    : dataset to train diffusion models
    """
    #
    ##
    def __init__(self, 
                 data_x):
        """
        <parameter>
        : data_x: array of samples, shape (samples, time, channels)
        """
        #
        ##
        super(DiffusionDataset, self).__init__()
        #
        data_x = np.swapaxes(data_x, -1, -2)
        #
        data_x_mean = np.expand_dims(np.mean(data_x, axis = -1), axis = -1)
        data_x_std = np.expand_dims(np.std(data_x, axis = -1), axis = -1)
        self.data_x = (data_x - data_x_mean) / data_x_std
        # self.data_x = data_x
        #
        self.var_num_sample = len(self.data_x)
    
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
        data_sample_x = self.data_x[var_i]
        #
        return data_sample_x










## ------------------------------------------------------------------------------------------ ##
## ----------------------------------- Mask Functions --------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
def mask_future(data_x,
                var_ratio):
    """
    <description>
    : function to mask the future steps of sequences to train models for forecasting
    <parameters>
    : data_x: batch of samples to mask, shape (samples, channels, time)
    : var_ratio: ratio to mask as future (lambda_fc)
    <return>
    : data_output_x: the masked samples, shape (samples, channels, time)
    """
    #
    ##
    var_len_mask = int( data_x.shape[-1] * var_ratio / (1.0 + var_ratio) )
    #
    var_mask = np.ones_like(data_x)
    #
    var_mask[..., -var_len_mask:] = 0
    #
    data_output_x = data_x * var_mask
    #
    ##
    return data_output_x

#
##
def pad_future(data_x,
               var_ratio):
    
    """
    <description>
    : function to lengthen samples for augmentation
    <parameters>
    : data_x: batch of samples to lengthen, shape (samples, channels, time)
    : var_ratio: ratio to lengthen as future (lambda_fc)
    <return>
    : data_output_x: the lengthened samples, shape (samples, channels, (1 + var_ratio) * time)
    """
    #
    ##
    var_len_future = int( data_x.shape[-1] * var_ratio )
    #
    var_shape_pad = list(data_x.shape)
    var_shape_pad[-1] = var_len_future + data_x.shape[-1]
    #
    data_output_x = np.zeros(var_shape_pad, dtype = np.float32)
    #
    data_output_x[..., :data_x.shape[-1]] = data_x
    #
    return data_output_x

#
##
def mask_random(data_x,
                var_ratio):
    
    """
    <description>
    : function to randomly mask values of samples to simulate missing values to train models for imputation
    <parameters>
    : data_x: batch of samples to mask, shape (samples, channels, time)
    : var_ratio: ratio to mask as missing values (lambda_miss/lambda_im)
    <return>
    : data_output_x: the masked samples, shape (samples, channels, time)
    """
    #
    ##
    var_num_value = 1
    #
    for var_shape in data_x.shape:
        #
        var_num_value = var_num_value * var_shape
    #
    var_mask = np.ones(var_num_value, dtype = np.float32)
    #
    var_num_zero = int(len(var_mask) * var_ratio)
    #
    var_index_zero = np.random.choice(len(var_mask), var_num_zero, replace = False)
    #
    var_mask[var_index_zero] = 0.0
    #
    var_mask = var_mask.reshape(data_x.shape)
    #
    data_output_x = torch.tensor(var_mask) * data_x
    #
    return data_output_x

#
##
def mask(data_x,
         var_mode,
         var_ratio_fc,
         var_ratio_im):
    
    """
    <description>
    : function to mask certain values of samples
    : to train models for forecasting and/or imputation
    <parameters>
    : data_x: batch of samples to mask, shape (samples, channels, time)
    : var_mode: masking model of forecasting and/or imputation
    : var_ratio_fc: ratio to mask as future (lambda_fc)
    : var_ratio_im: ratio to mask as missing values (lambda_miss/lambda_im)
    <return>
    : data_output_x: the masked samples, shape (samples, channels, time)
    """
    #
    ##
    if var_mode == "forecast":
        return mask_future(data_x, var_ratio_fc)
    #
    elif var_mode == "imputation":
        return mask_random(data_x, var_ratio_im)
    #
    elif var_mode == "mix":
        return mask_future(mask_random(data_x, var_ratio_im), var_ratio_fc)
    #
    else:
        raise KeyError("Mode of mask should be \"forecast\", \"imputation\" or \"mix\".")










## ------------------------------------------------------------------------------------------ ##
## ----------------------------------------- Test ------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
def test_mask_random():
    #
    ##
    data_x = np.load(preset["path"]["data"])
    # 
    data_sample_x = np.swapaxes(data_x[0:16], -1, -2)
    #
    data_mask_x = mask_random(data_sample_x, 0.2)
    #
    print(data_mask_x.shape)










## ------------------------------------------------------------------------------------------ ##
## ----------------------------------------- Main ------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
if __name__ == "__main__":
    #
    ##
    test_mask_random()