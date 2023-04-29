"""
[file]          metrics.py
[description]   functions to support metric calculation
"""
#
##
import pandas as pd
from sklearn.metrics import *

#
##
def create_result_dataframe(var_true_y, 
                            var_predict_y, 
                            var_true_proba = None, 
                            var_predict_proba = None, 
                            var_labels = None):
    """
    <description>
    : function to record the ground-true labels and predicted labels
    : the record can generate detailed metrics (e.g., accuracy, precision, recall, F1, confusion matrix)
    <parameters>
    : var_true_y:           array of ground-true labels, shape: (samples,)
    : var_predict_y:        array of predicted labels, shape: (samples,)
    : var_true_proba:       the encoded labels, which are usually one-hot encoding
    :                       shape: (samples, encoded features)
    : var_predict_proba:    the probabilities of predicted labels, which are usually the output of softmax
    :                       shape: (samples, encoded features)
    : var_labels:           array of labels
    <return>
    : var_result: dataframe recording the ground-true labels and predicted labels
    """
    #
    ## ground-true labels
    var_true_y_df = pd.DataFrame(var_true_y, columns = ["True"])
    #
    ## predicted labels
    var_predict_y_df = pd.DataFrame(var_predict_y, columns = ["Predict"])
    #
    ##
    var_labels = var_labels if var_labels is not None else list(set(var_true_y))
    #
    ## encodings (probabilities) of labels
    var_true_proba_df = None
    if var_true_proba is not None:
        var_true_proba_df = pd.DataFrame(var_true_proba, 
            columns = ["True_" + var_label for var_label in var_labels])
    #
    ## probabilities of predicted labels
    var_predict_proba_df = None
    if var_predict_proba is not None:
        var_predict_proba_df = pd.DataFrame(var_predict_proba, 
            columns = ["Predict_" + var_label for var_label in var_labels])
    #
    ##
    var_result = pd.concat([var_true_y_df, var_true_proba_df, 
                            var_predict_y_df, var_predict_proba_df], axis = 1)
    #
    return var_result