"""
[file]          metrics.py
[description]   calculate metrics to measure the quality of augmented CSI
"""
#
##
import torch
import numpy as np










## ------------------------------------------------------------------------------------------ ##
## ---------------------------------------- Metrics ----------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
def calc_mse(var_predict, var_target):
    """
    <description>
    : function to calculate MSE between "var_predict" and "var_target"
    <parameters>
    : var_predict: one predicted (synthesized) sample
    : var_target: one ground-true sample
    <return>
    : var_mse: MSE between "var_forecast" and "var_target"
    """
    #
    ##
    with torch.no_grad():
        #
        var_predict = torch.tensor(var_predict)
        var_target = torch.tensor(var_target)
        #
        var_mse = torch.nn.functional.mse_loss(var_predict, var_target)
    #
    var_mse = var_mse.cpu().numpy()
    return var_mse

#
##
def calc_mae(var_predict, var_target):
    """
    <description>
    : function to calculate MAE between "var_predict" and "var_target"
    <parameters>
    : var_predict: one predicted (synthesized) sample
    : var_target: one ground-true sample
    <return>
    : var_mae: MAE between "var_forecast" and "var_target"
    """
    #
    ##
    with torch.no_grad():
        #
        var_predict = torch.tensor(var_predict)
        var_target = torch.tensor(var_target)
        #
        var_mae = torch.nn.functional.l1_loss(var_predict, var_target)
    #
    var_mae = var_mae.cpu().numpy()
    return var_mae

#
##
def calc_quantile_loss(var_predict, var_target, var_q):
    """
    <description>
    : function to calculate quantile loss between "var_predict" and "var_target"
    <parameters>
    : var_predict: one predicted (synthesized) sample
    : var_target: one ground-true sample
    : var_q: the quantile
    <return>
    : var_quantile_loss: quantile loss between "var_forecast" and "var_target"
    """
    #
    ##
    var_quantile_loss = (var_predict - var_target) * ((var_target <= var_predict) * 1.0 - var_q)
    var_quantile_loss = 2 * torch.sum(torch.abs(var_quantile_loss))
    #
    return var_quantile_loss

#
##
def calc_crps(var_predict, var_target):
    """
    <description>
    : function to calculate CRPS between "var_predict" and "var_target"
    <parameters>
    : var_predict: one predicted (synthesized) sample
    : var_target: one ground-true sample
    <return>
    : var_crps_avg: CRPS between "var_forecast" and "var_target"
    """
    #
    ##
    var_quantile = (np.arange(10)/10.0)[1:]
    #
    var_crps_sum = 0.0
    #
    for var_q in var_quantile:
        #
        var_crps_sum += calc_quantile_loss(var_target, var_predict, var_q)
    #
    var_crps_avg = var_crps_sum / len(var_quantile)
    #
    var_crps_avg = var_crps_avg / (torch.sum(abs(var_target)))
    #
    return var_crps_avg
