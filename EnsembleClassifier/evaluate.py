"""
[file]          evaluate.py
[description]   evaluate DiffAR by feeding incomplete data and augmented data to the ensemble classifier
"""
#
##
import json
import numpy as np
import datetime, time
import torch
#
from metrics import *
from preset  import *
from preprocess import *
from model import *

#
##
print("-" * 39)
print(" Experiment", datetime.datetime.now())
print("-" * 39)

#
##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
print(preset)










## ------------------------------------------------------------------------------------------ ##
## ------------------------------------------ Data ------------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
#
##
var_index = json.load(open(preset["path"]["index"]))
#
##
data_x = np.load(preset["path"]["data_incomplete_x"], allow_pickle = True)
data_aug_x = np.load(preset["path"]["data_augment_x"], allow_pickle = True)
data_y = np.load(preset["path"]["data_y"], allow_pickle = True)
#
print(data_x.shape, data_aug_x.shape, data_y.shape)
#
##
data_train_set, data_val_set, data_test_set, var_label_encoder = \
    preprocess(data_x, data_aug_x, data_y, var_index)
#
data_train_loader = torch.utils.data.DataLoader(dataset = data_train_set, shuffle = True,
                                                batch_size = preset["train"]["batch_size"])
#
data_val_loader = torch.utils.data.DataLoader(dataset = data_val_set, shuffle = False,
                                              batch_size = preset["train"]["batch_size"])
#
data_test_loader = torch.utils.data.DataLoader(dataset = data_test_set, shuffle = False,
                                               batch_size = preset["train"]["batch_size"])











## ------------------------------------------------------------------------------------------ ##
## ----------------------------------------- Model ------------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
#
##
model_classifer = EnsembleClassifer(var_dim_input = data_aug_x[0].shape[0], 
                                    var_dim_output = len(set(data_y)),
                                    **preset["classifier"])
#
model_classifer = model_classifer.to(device)
#
##
optimizer = torch.optim.Adam(model_classifer.parameters(), lr = preset["train"]["lr"])
#
##
loss = torch.nn.CrossEntropyLoss(label_smoothing = 0.1)
#
##
var_path_initial = preset["path"]["save"] + "classifier_initial.pt"
#
torch.save(model_classifer.state_dict(), var_path_initial)










## ------------------------------------------------------------------------------------------ ##
## --------------------------------------- Evaluate ----------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
var_repeat = 10    # times to repeat experiments
#
for var_r in range(var_repeat):
    #
    ##
    ## ------------------------------------- Model ------------------------------------------ ##
    #
    ##
    model_classifer.load_state_dict(torch.load(var_path_initial))
    #
    var_path_best = preset["path"]["save"] + "classifier_" + str(var_r) + ".pt"
    #
    var_acc_best = 0.0
    #
    ##
    ## ------------------------------------- Train ------------------------------------------ ##
    #
    ##
    var_epochs = preset["train"]["epochs"]
    #
    ##
    for var_epoch in range(var_epochs):
        #
        ##
        var_num_train = 0
        var_correct_train = 0
        #
        model_classifer.train()
        var_time_0 = time.time()
        #
        ##
        for data_batch in data_train_loader:
            #
            ##
            data_x, data_aug_x, data_y = [var.to(device) for var in data_batch]
            #
            predict_logit_y = model_classifer(data_x, data_aug_x)
            #
            var_loss_train = loss(predict_logit_y, data_y)
            #
            optimizer.zero_grad()
            #
            var_loss_train.backward()
            #
            torch.nn.utils.clip_grad_norm_(model_classifer.parameters(), 1.0)
            #
            optimizer.step()
            #
            ##
            predict_y = torch.max(predict_logit_y, 1)[1]
            var_num_train += data_y.cpu().shape[0]
            var_correct_train += (predict_y.cpu() == data_y.cpu()).sum()
        #
        ##
        ## --------------------------------- Validation -------------------------------------- ##
        #
        ##
        var_num_val = 0
        var_correct_val = 0
        #
        model_classifer.eval()
        #
        with torch.no_grad():
            #
            for data_batch in data_val_loader:
                #
                ##
                data_x, data_aug_x, data_y = [var.to(device) for var in data_batch]
                #
                predict_logit_y = model_classifer(data_x, data_aug_x)
                #
                var_loss_val = loss(predict_logit_y, data_y)
                #
                ##
                predict_y = torch.max(predict_logit_y, 1)[1]
                #
                var_num_val += data_y.cpu().shape[0]
                var_correct_val += (predict_y.cpu() == data_y.cpu()).sum()
        #
        ##
        var_time_1 = time.time()
        #
        print(f"Epoch {var_epoch}/{var_epochs}",
            "- Time %.6fs"%(var_time_1 - var_time_0),
            "- Loss %.6f"%var_loss_train.cpu(),
            "- Accuracy %.6f"%(var_correct_train / var_num_train),
            "- Val Loss %.6f"%var_loss_val.cpu(),
            "- Val Accuracy %.6f"%(var_correct_val / var_num_val))
        #
        ##
        if var_acc_best < (var_correct_val / var_num_val):
            
            var_acc_best = (var_correct_val / var_num_val)
            
            torch.save(model_classifer.state_dict(), var_path_best)
    #
    ##
    ## ------------------------------------- Test ------------------------------------------ ##
    #
    ##
    data_test_y = []
    predict_test_y = []
    predict_test_y_p = []
    #
    ##
    model_classifer.load_state_dict(torch.load(var_path_best))
    model_classifer.eval()
    #
    with torch.no_grad():
        #
        for data_batch in data_test_loader:
            #
            ##
            data_x, data_aug_x, data_y = [var.to(device) for var in data_batch]
            #
            predict_logit_y = model_classifer(data_x, data_aug_x)
            #
            predict_y = torch.max(predict_logit_y, 1)[1].cpu().numpy()
            predict_y = var_label_encoder.inverse_transform(predict_y)
            #
            predict_test_y.append(predict_y)
            predict_test_y_p.append(predict_logit_y.cpu().numpy())
            #
            data_y = var_label_encoder.inverse_transform(data_y.cpu().numpy())
            data_test_y.append(data_y)
    #
    ##
    data_test_y = np.concatenate(data_test_y)
    predict_test_y = np.concatenate(predict_test_y)
    predict_test_y_p = np.concatenate(predict_test_y_p)
    #
    print("Repeat %d - validation acc - %0.6f - test acc - %0.6f" % \
        (var_r, var_acc_best, accuracy_score(data_test_y, predict_test_y)))
    #
    print("Report", var_r)
    print(classification_report(data_test_y, predict_test_y, output_dict = True))
    #
    ##
    result_test = create_result_dataframe(var_true_y = data_test_y, 
                                        var_predict_y = predict_test_y, 
                                        var_predict_proba = predict_test_y_p, 
                                        var_labels = var_label_encoder.classes_)
    #
    result_test.to_csv(preset["path"]["save"] + "result_test_" + str(var_r) + ".csv")