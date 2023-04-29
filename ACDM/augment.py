"""
[file]          augment.py
[description]   augment the entire dataset for the subsequent training of ensemble classifier
"""
#
##
import json
import numpy as np
import datetime, time
import torch, torchaudio
#
from preprocess import *
from model import *
from preset import *
from predict import *
from metrics import *

#
##
print("-" * 41)
print(" Augmentation", datetime.datetime.now())
print("-" * 41)
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
## --------------------------------------- Augment ------------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
#
##
data_incomplete_x = []
data_aug_x = []
#
##
for var_i, data_sample_x in enumerate(data_x):
    #
    ##
    var_time_0 = time.time()
    #
    ## data_sample_x, shape (time, features)
    data_sample_x = data_sample_x.reshape(data_sample_x.shape[0], -1)
    data_sample_x = np.expand_dims(np.swapaxes(data_sample_x, -1, -2), axis = 0)
    #
    data_x_mean = np.expand_dims(np.mean(data_sample_x, axis = -1), axis = -1)
    data_x_std = np.expand_dims(np.std(data_sample_x, axis = -1), axis = -1)
    data_sample_x = (data_sample_x - data_x_mean) / data_x_std
    #
    data_sample_x = torch.tensor(data_sample_x)
    #
    ## random mask to simulate missing values
    data_mask_x = mask_random(data_sample_x, preset["train"]["ratio_im"])
    #
    data_incomplete_x.append(data_mask_x.detach().cpu().numpy())
    #
    ## lengthen data for forecasting
    data_mask_x = pad_future(data_mask_x, preset["train"]["ratio_fc"])
    #
    data_mask_x = torch.tensor(data_mask_x)
    # print(data_mask_x.shape)
    #
    ##
    data_result_x = predict(model_acdm, data_mask_x).detach().cpu().numpy()
    #
    var_time_1 = time.time()
    #
    var_time = var_time_1 - var_time_0
    #
    ##
    print(f"Sample {var_i}", 
            "- Time %.6f"%var_time
    )
    #
    ##
    data_aug_x.append(data_result_x)
#
##
data_incomplete_x = np.concatenate(data_incomplete_x, dtype = object)
data_aug_x = np.concatenate(data_aug_x, dtype = object)
#
## save the incomplete samples for evaluation
with open(preset["path"]["save"] + "data_incomplete_x.npy", "wb") as var_file:
    #
    np.save(var_file, data_incomplete_x)
#
## save the augmented samples for evaluation
with open(preset["path"]["save"] + "data_augment_x.npy", "wb") as var_file:
    #
    np.save(var_file, data_aug_x)