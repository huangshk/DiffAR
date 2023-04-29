"""
[file]          model.py
[description]   implement the ensemble classifier
"""
#
##
import torch










## ------------------------------------------------------------------------------------------ ##
## ---------------------------------- Ensemble Classifer ------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
#
##
class EnsembleClassifer(torch.nn.Module):
    """
    <description>
    : module of ensemble classifier
    """
    #
    ##
    def __init__(self,
                 var_dim_input,
                 var_dim_output,
                 var_dim_model,
                 var_num_head,
                 var_dim_forward,
                 var_num_layer):
        #
        ##
        super(EnsembleClassifer, self).__init__()
        #
        ##
        self.layer_cnn_0 = torch.nn.ModuleList([
            #
            torch.nn.Conv1d(var_dim_input, var_dim_model//4, 7, stride = 3),
            # torch.nn.BatchNorm1d(var_dim_model//4),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            #
            torch.nn.Conv1d(var_dim_model//4, var_dim_model//2, 5, stride = 2),
            # torch.nn.BatchNorm1d(var_dim_model//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            #
            torch.nn.Conv1d(var_dim_model//2, var_dim_model, 3, stride = 1),
            # torch.nn.BatchNorm1d(var_dim_model),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
        ])
        #
        for layer in self.layer_cnn_0: 
            if isinstance(layer, torch.nn.Conv1d): 
                torch.nn.init.xavier_uniform_(layer.weight)
        #
        ##
        self.layer_cnn_1 = torch.nn.ModuleList([
            #
            torch.nn.Conv1d(var_dim_input, var_dim_model//4, 7, stride = 3),
            # torch.nn.BatchNorm1d(var_dim_model//4),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            #
            torch.nn.Conv1d(var_dim_model//4, var_dim_model//2, 5, stride = 2),
            # torch.nn.BatchNorm1d(var_dim_model//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            #
            torch.nn.Conv1d(var_dim_model//2, var_dim_model, 3, stride = 1),
            # torch.nn.BatchNorm1d(var_dim_model),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
        ])
        #
        for layer in self.layer_cnn_1: 
            if isinstance(layer, torch.nn.Conv1d): 
                torch.nn.init.xavier_uniform_(layer.weight)
        #
        ##
        layer_encoder = torch.nn.TransformerEncoderLayer(
            var_dim_model * 2, 
            var_num_head, 
            var_dim_forward, 
            batch_first = True, 
            norm_first = True
        )
        self.layer_encoder = torch.nn.TransformerEncoder(layer_encoder, var_num_layer)
        #
        ##
        self.layer_output = torch.nn.Linear(var_dim_model * 2, var_dim_output)
        torch.nn.init.xavier_uniform_(self.layer_output.weight)

    #
    ##
    def forward(self, 
                var_input_0,
                var_input_1):
        """
        : var_input_0: shape (batch_size, var_dim_input, var_len_sequence)
        : var_input_1: shape (batch_size, var_dim_input, var_len_sequence)
        """
        #
        ##
        var_x_0 = var_input_0
        var_x_1 = var_input_1
        #
        ## CNN-1D
        for layer in self.layer_cnn_0: var_x_0 = layer(var_x_0)
        for layer in self.layer_cnn_1: var_x_1 = layer(var_x_1)
        #
        ##
        var_x = torch.cat((var_x_0, var_x_1), dim = 1)
        #
        ## (batch_size, var_len_sequence, var_dim_model * 2)
        var_x = torch.permute(var_x, (0, 2, 1))
        #
        ## Transformer Encoder
        var_x = self.layer_encoder(var_x)
        #
        ##
        var_x = torch.permute(var_x, (0, 2, 1))
        #
        ##
        var_x = torch.mean(var_x, dim = -1)
        #
        ## Linear
        var_x = self.layer_output(var_x)
        #
        var_output = var_x
        #
        return var_output

