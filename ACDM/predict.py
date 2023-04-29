"""
[file]          predict.py
[description]   predict (synthesize) new (augmented) samples using ACDM
"""
#
##
import numpy as np
import datetime, time
import torch, torchaudio
#
from preprocess import mask
from model import ACDM
from preset import preset
from metrics import *










## ------------------------------------------------------------------------------------------ ##
## ---------------------------------------- Predict ----------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
def predict(model, 
            data_batch_x):
    """
    <description>
    : function to synthesize new (augmented) samples from "data_batch_x" using "model"
    <parameters>
    : model: trained ACDM model to synthesize new samples
    : data_batch_x: raw samples to augment
    <return>
    : data_output_x: synthesized samples
    """
    #
    ##
    device = next(model.parameters()).device
    #
    var_beta = np.linspace(preset["train"]["min_beta"], 
                           preset["train"]["max_beta"], 
                           preset["train"]["max_step"])
    var_alpha = 1 - var_beta
    var_alpha_bar = np.cumprod(var_alpha)
    var_alpha_bar = torch.tensor(var_alpha_bar.astype(np.float32)).to(device)
    #
    ##
    with torch.no_grad():
        #
        ## using spectrogram as input conditions
        data_spectrogram = torchaudio.transforms.Spectrogram(
            **preset["spectrogram"])(data_batch_x).to(device)
        #
        # data_spectrogram = 10 * torch.log10(data_spectrogram + 1e-12)
        #
        ## x^T = Gaussian Noise
        var_x = torch.randn_like(data_batch_x).to(device)
        #
        ## for t = [T, ..., 1]
        for var_t in range(preset["train"]["max_step"] - 1, -1, -1):
            #
            var_y = model(var_x, var_t, data_spectrogram)
            #
            ## line 7 of algorithm 2 in the paper
            var_x = var_alpha[var_t] ** (-0.5) * (var_x - var_beta[var_t] * (1.0 - var_alpha_bar[var_t]) ** (-0.5) * var_y)
            #
            if var_t > 0:
                #
                var_z = torch.randn_like(var_x)
                var_sigma = ( (1.0 - var_alpha_bar[var_t - 1]) / (1.0 - var_alpha_bar[var_t]) * var_beta[var_t] )**0.5
                # var_sigma = ( var_beta[var_t] )**0.5
                var_x += var_sigma * var_z
                #
            var_x = torch.clamp(var_x, -1.0, 1.0)
    #
    ##
    data_output_x = var_x
    #
    return data_output_x










## ------------------------------------------------------------------------------------------ ##
## ----------------------------------------- Test ------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
def test_predict(var_index):
    """
    <description>
    : function to synthesize one new (augmented) sample of "var_index" for testing
    <parameters>
    : var_index: index of the sample to synthesize (augment) in dataset
    <return>
    : None
    """
    #
    ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    print(preset)
    #
    ##
    data_x = np.load(preset["path"]["data"], allow_pickle = True)
    # 
    data_sample_x = data_x[var_index]
    data_sample_x = data_sample_x.reshape(data_sample_x.shape[0], -1)
    data_sample_x = np.expand_dims(np.swapaxes(data_sample_x, -1, -2), axis = 0)
    #
    data_x_mean = np.expand_dims(np.mean(data_sample_x, axis = -1), axis = -1)
    data_x_std = np.expand_dims(np.std(data_sample_x, axis = -1), axis = -1)
    data_sample_x = (data_sample_x - data_x_mean) / data_x_std
    #
    data_sample_x = torch.tensor(data_sample_x)
    #
    ## mask raw data for testing
    data_mask_x = mask(data_sample_x, 
                       var_mode = preset["train"]["mode"], 
                       var_ratio_fc = preset["train"]["ratio_fc"], 
                       var_ratio_im = preset["train"]["ratio_im"])
    #
    ##
    model_acdm = ACDM(var_dim_condition = preset["spectrogram"]["n_fft"] // 2 + 1,
                      **preset["acdm"]).to(device)
    #
    model_acdm.load_state_dict(torch.load(preset["path"]["model"]))
    #
    model_acdm.eval()
    #
    ##
    data_result_x = predict(model_acdm, data_mask_x).detach().cpu()
    data_sample_x = data_sample_x.detach().cpu()
    #
    var_mae = calc_mae(data_result_x, data_sample_x)
    var_mse = calc_mse(data_result_x, data_sample_x)
    var_crps = calc_crps(data_result_x, data_sample_x)
    #
    data_result_x = data_result_x * data_x_std + data_x_mean
    data_result_x = data_result_x.squeeze(0)
    # 
    print(data_result_x.shape)
    # 
    print(f"MAE: {var_mae}", f"MSE: {var_mse}", f"CRPS: {var_crps}")
    #
    ##
    with open(preset["path"]["save"] + "data_x_ag.npy", "wb") as var_file:
        #
        np.save(var_file, data_result_x)










## ------------------------------------------------------------------------------------------ ##
## ----------------------------------------- Main ------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
if __name__ == "__main__":
    #
    ##
    test_predict(0)