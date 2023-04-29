"""
[file]          evaluate.py
[description]   evaluate the quality of augmented CSI
"""
#
##
import json
import numpy as np
import datetime, time
import torch, torchaudio
#
from preprocess import DiffusionDataset, mask
from model import ACDM
from preset import preset
from predict import *
from metrics import *

#
##
print("-" * 39)
print(" Experiment", datetime.datetime.now())
print("-" * 39)
#
print(preset)

#
##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")










## ------------------------------------------------------------------------------------------ ##
## ------------------------------------------ Data ------------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
#
##
data_x = np.load(preset["path"]["data"], allow_pickle = True)
#
var_index = json.load(open(preset["path"]["index"]))
#
## use test set for evaluation
data_test_x = data_x[var_index["test"]]
                                                 









## ------------------------------------------------------------------------------------------ ##
## ----------------------------------------- Model ------------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
#
##
model_acdm = ACDM(var_dim_condition = preset["spectrogram"]["n_fft"] // 2 + 1,
                  **preset["acdm"]).to(device)
#
model_acdm.load_state_dict(torch.load(preset["path"]["model"]))
#
model_acdm.eval()










## ------------------------------------------------------------------------------------------ ##
## --------------------------------------- Evaluate ----------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
var_sum_mae = 0
var_sum_mse = 0
var_sum_crps = 0
#
##
for var_i, data_sample_x in enumerate(data_test_x):
    #
    ##
    var_time_0 = time.time()
    #
    ## data_sample_x (time, features)
    data_sample_x = data_sample_x.reshape(data_sample_x.shape[0], -1)
    data_sample_x = np.expand_dims(np.swapaxes(data_sample_x, -1, -2), axis = 0)
    #
    data_x_mean = np.expand_dims(np.mean(data_sample_x, axis = -1), axis = -1)
    data_x_std = np.expand_dims(np.std(data_sample_x, axis = -1), axis = -1)
    data_sample_x = (data_sample_x - data_x_mean) / data_x_std
    #
    data_sample_x = torch.tensor(data_sample_x)
    #
    ## mask raw data for evaluation
    data_mask_x = mask(data_sample_x, 
                       var_mode = preset["train"]["mode"], 
                       var_ratio_fc = preset["train"]["ratio_fc"], 
                       var_ratio_im = preset["train"]["ratio_im"])
    #
    ##
    data_result_x = predict(model_acdm, data_mask_x).detach().cpu()
    data_sample_x = data_sample_x.detach().cpu()
    #
    var_mae = calc_mae(data_result_x, data_sample_x)
    var_mse = calc_mse(data_result_x, data_sample_x)
    var_crps = calc_crps(data_result_x, data_sample_x)
    #
    var_time_1 = time.time()
    #
    var_time = var_time_1 - var_time_0
    #
    ##
    print(f"Sample {var_i}", 
            "- Time %.6f"%var_time, 
            "- MAE %.6f"%var_mae, 
            "- MSE %.6f"%var_mse, 
            "- CRPS %.6f"%var_crps
    )
    #
    var_sum_mae += var_mae
    var_sum_mse += var_mse
    var_sum_crps += var_crps
#
##
var_avg_mae = var_sum_mae / len(data_test_x)
var_avg_mse = var_sum_mse / len(data_test_x)
var_avg_crps = var_sum_crps / len(data_test_x)
#
print("Average",
      "- MAE %.6f"%var_avg_mae, 
      "- MSE %.6f"%var_avg_mse, 
      "- CRPS %.6f"%var_avg_crps
)