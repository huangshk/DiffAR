"""
[file]          train.py
[description]   train ACDM in a self-supervised manner
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
#
var_new_model = True










## ------------------------------------------------------------------------------------------ ##
## ------------------------------------------ Data ------------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
#
##
data_x = np.load(preset["path"]["data"], allow_pickle = True)
#
var_index = json.load(open(preset["path"]["index"]))
#
data_train_x = data_x[var_index["train"]]
#
data_train_x_set = DiffusionDataset(data_train_x)
#
data_train_x_loader = torch.utils.data.DataLoader(dataset = data_train_x_set,
                                                  batch_size = preset["train"]["batch_size"],
                                                  shuffle = True,
                                                  num_workers = 4)










## ------------------------------------------------------------------------------------------ ##
## ----------------------------------------- Model ------------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
#
##
model_acdm = ACDM(var_dim_condition = preset["spectrogram"]["n_fft"] // 2 + 1,
                  **preset["acdm"]).to(device)
#
if (preset["path"]["model"] is not None) and (not var_new_model):
    model_acdm.load_state_dict(torch.load(preset["path"]["model"]))
#
model_acdm.train()
#
torch.backends.cudnn.benchmark = True
#
var_optimizer = torch.optim.Adam(model_acdm.parameters(), lr = preset["train"]["lr"])
#
## linear spaced noise schedule
var_beta = np.linspace(preset["train"]["min_beta"], 
                       preset["train"]["max_beta"], 
                       preset["train"]["max_step"])
var_alpha = 1 - var_beta
var_alpha_bar = np.cumprod(var_alpha)
var_alpha_bar = torch.tensor(var_alpha_bar.astype(np.float32)).to(device)










## ------------------------------------------------------------------------------------------ ##
## ----------------------------------------- Train ------------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
#
##
for var_epoch in range(preset["train"]["epochs"]):
    #
    ##
    var_time_0 = time.time()
    #
    var_loss_avg = 0
    #
    ##
    for var_batch, data_batch_x in enumerate(data_train_x_loader):
        #
        ## mask raw data for training
        data_mask_x = mask(data_batch_x, 
                           var_mode = preset["train"]["mode"], 
                           var_ratio_fc = preset["train"]["ratio_fc"], 
                           var_ratio_im = preset["train"]["ratio_im"])
        #
        ## using spectrogram as input conditions
        data_spectrogram = torchaudio.transforms.Spectrogram(
            **preset["spectrogram"])(data_mask_x).to(device)
        #
        # data_spectrogram = 10 * torch.log10(data_spectrogram + 1e-12)
        #
        data_batch_x = data_batch_x.to(device)
        #
        ## different diffusion steps for different samples in a batch
        var_t = torch.randint(0, preset["train"]["max_step"], [data_batch_x.shape[0]]).to(device)
        #
        ## different alpha_bar at different diffusion steps for different samples in a batch
        var_alpha_bar_t = var_alpha_bar[var_t].unsqueeze(-1).unsqueeze(-1)
        #
        ## noise
        var_epsilon = torch.randn_like(data_batch_x)
        #
        ## line 7 of algorithm 1 in the paper
        var_x_t = var_alpha_bar_t**0.5 * data_batch_x + (1.0 - var_alpha_bar_t)**0.5 * var_epsilon
        #
        var_y = model_acdm(var_x_t, var_t, data_spectrogram)
        #
        ## gradient step
        var_optimizer.zero_grad()
        #
        var_loss = torch.nn.functional.mse_loss(var_epsilon, var_y)
        #
        var_loss.backward()
        #
        var_loss_avg += var_loss.item()
        #
        torch.nn.utils.clip_grad_norm_(model_acdm.parameters(), 1e9)
        #
        var_optimizer.step()
    #
    ##
    print("Epoch %d/%d" % (var_epoch, preset["train"]["epochs"]),
          "- Time %.4fs" % (time.time() - var_time_0),
          "- Loss %.4f" % (var_loss_avg / (var_batch + 1)))
    #
    ## save model
    if (var_epoch) % 50 == 0:
        #
        var_path_save = preset["path"]["save"] + "weight_" + str(var_epoch) + ".pt"
        torch.save(model_acdm.state_dict(), var_path_save)
