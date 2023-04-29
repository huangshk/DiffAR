"""
[file]          model.py
[description]   implement Adaptive Conditional Diffusion Model (ACDM)
"""
#
##
import torch
import torchaudio
import numpy as np
#
from preset import preset










## ------------------------------------------------------------------------------------------ ##
## ------------------------------------ Step Embedding -------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
class StepEmbedding(torch.nn.Module):
    """
    <description>
    : module to embed diffusion steps
    """
    #
    ##
    def __init__(self,
                 var_dim_step,
                 var_max_step):
        """
        <parameter>
        : var_dim_step: dimension of embedded diffusion steps
        : var_max_step: total number of diffusion steps
        """
        #
        ##
        super(StepEmbedding, self).__init__()
        #
        var_step_array = torch.arange(var_max_step).unsqueeze(1)
        # (var_max_step, 1)
        #
        var_encoding_array = torch.arange(int(var_dim_step // 2)).unsqueeze(0)    
        # (1, var_dim_step//2)
        #
        var_encoding_array = 10.0 ** (var_encoding_array * 4 / (var_dim_step // 2 - 1)) 
        # (1, var_dim_step//2)
        #
        var_encoding = var_step_array * var_encoding_array 
        # (var_max_step, var_dim_step//2)
        #
        var_encoding = torch.cat([torch.sin(var_encoding), torch.cos(var_encoding)], dim = 1)
        # (var_max_step, var_dim_step)
        #
        self.var_encoding = torch.nn.Parameter(var_encoding, requires_grad = False)
        #
        self.layer_project_0 = torch.nn.Linear(var_dim_step, var_dim_step)
        #
        self.layer_project_1 = torch.nn.Linear(var_dim_step, var_dim_step)
    
    #
    ##
    def forward(self, 
                var_step):
        """
        <parameter>
        : var_step: batch of diffusion steps, shape (batch_size, )
        <return>
        : var_output: batch of embedded diffusion steps, shape (batch_size, var_dim_step)
        """
        #
        ##
        var_x = var_step
        #
        var_x = self.var_encoding[var_x]
        #
        var_x = self.layer_project_0(var_x)
        # var_x = torch.nn.functional.silu(var_x)
        #
        var_x = self.layer_project_1(var_x)
        # var_output = torch.nn.functional.silu(var_x)
        #
        var_output = var_x
        #
        return var_output










## ------------------------------------------------------------------------------------------ ##
## --------------------------------- Adaptive Conditioner ----------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
class AdaptiveConditioner(torch.nn.Module):
    """
    <description>
    : module to learn step-specific conditions from spectrogram according to diffusion steps
    """
    #
    ##
    def __init__(self,
                 var_dim_feature,
                 var_scale,
                 var_dim_condition,
                 var_dim_step):
        """
        <parameter>
        : var_dim_feature: dimension (CSI channels) of input tensors
        : var_scale: scale to upsample the time axis of spectrogram
        : var_dim_condition: dimension of input condition (spectrogram)
        : var_dim_step: dimension of encoded diffusion steps
        """
        #
        ##
        super(AdaptiveConditioner, self).__init__()
        #
        ##
        var_stride = int(var_scale ** 0.5)
        #
        ## deconvolutional layers (Deconv)
        self.layer_upsample_0 = torch.nn.ConvTranspose2d(var_dim_feature,     
                                                         var_dim_feature, 
                                                         kernel_size = (3, var_stride * 2), 
                                                         stride = (1, var_stride), 
                                                         padding = (1, var_stride // 2))
        #
        self.layer_upsample_1 = torch.nn.ConvTranspose2d(var_dim_feature, 
                                                         var_dim_feature, 
                                                         kernel_size = (3, var_stride * 2), 
                                                         stride = (1, var_stride), 
                                                         padding = (1, var_stride // 2))
        #
        self.layer_filter = torch.nn.Linear(var_dim_step, var_dim_condition)

    #
    ##
    def forward(self,
                var_spectrogram,
                var_step):
        """
        <parameter>
        : var_spectrogram: batch of spectrogram, shape (batch_size, var_dim_feature, var_dim_condition, time_short)
        : var_step: batch of encoded diffusion steps, shape (batch_size, var_dim_step)
        <return>
        : var_output: shape (batch_size, var_dim_condition, var_dim_feature, time)
        """
        #
        ##
        var_x = var_spectrogram
        #
        var_x = self.layer_upsample_0(var_x)
        # var_x = torch.nn.functional.leaky_relu(var_x, 0.1)
        # var_x = torch.relu(var_x)
        #
        var_x = self.layer_upsample_1(var_x)
        # var_x = torch.nn.functional.leaky_relu(var_x, 0.1)
        # var_x = torch.relu(var_x)
        #
        var_output = torch.permute(var_x, (0, 2, 1, 3))
        #
        ## step-specific filter
        var_filter = self.layer_filter(var_step).unsqueeze(-1).unsqueeze(-1)
        #
        ## element-wise multiplication
        var_output = var_output * torch.sigmoid(var_filter)
        # 
        return var_output










## ------------------------------------------------------------------------------------------ ##
## ---------------------------- Multi-scale Dilated Convolution ----------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
class MultiScaleDilatedConv(torch.nn.Module):
    """
    <description>
    : module to perform multi-scale dilated convolution
    """
    #
    ##
    def __init__(self,
                 var_dim_in,
                 var_dim_out,
                 var_conv_size_list,
                 var_conv_dilation):
        """
        <parameter>
        : var_dim_in: dimension of input
        : var_dim_out: dimension of output
        : var_conv_size_list: list of convolutional kernel sizes
        : var_conv_dilation: length of convolutional dilation
        """
        #
        ##
        super(MultiScaleDilatedConv, self).__init__()
        #
        ##
        layer_conv_list = []
        #
        for var_conv_size in var_conv_size_list:
            #
            layer_conv = torch.nn.Conv2d(in_channels = var_dim_in, 
                                         out_channels = var_dim_out, 
                                         dilation = (1, var_conv_dilation),
                                         kernel_size = (1, var_conv_size),
                                         padding = "same")
            torch.nn.init.kaiming_normal_(layer_conv.weight)
            #
            layer_conv_list.append(layer_conv)
        #
        self.layer_conv_list = torch.nn.ModuleList(layer_conv_list)

    #
    ##
    def forward(self, 
                var_input):
        """
        <parameter>
        : var_input: batch of tensors, shape (batch_size, var_dim_in, var_dim_feature, time)
        <return>
        : var_output: batch of tensors, shape (batch_size, var_dim_out, var_dim_feature, time)
        """
        #
        ##
        var_x = var_input
        var_y = None
        #
        ## multi-scale convolution
        for layer_conv in self.layer_conv_list:
            #
            var_y = layer_conv(var_x) if var_y is None else var_y + layer_conv(var_x)
        #
        var_output = var_y / len(self.layer_conv_list) # ** 0.5
        #
        return var_output










## ------------------------------------------------------------------------------------------ ##
## ----------------------------------- Residual Block --------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
class ResidualBlock(torch.nn.Module):
    """
    <description>
    : module of residual block
    """
    #
    ##
    def __init__(self,
                 var_dim_residual,
                 var_dim_step,
                 var_dim_condition,
                 var_conv_size_list,
                 var_conv_dilation):
        """
        <parameter>
        : var_dim_residual: dimension of residual block for residual connection
        : var_dim_step: dimension of encoded diffusion steps
        : var_dim_condition: dimension of input condition (spectrogram)
        : var_conv_size_list: list of convolutional kernel sizes
        : var_conv_dilation: length of convolutional dilation
        """
        #
        ##
        super(ResidualBlock, self).__init__()
        #
        ##
        self.layer_step = torch.nn.Linear(var_dim_step, var_dim_residual)
        #
        self.layer_condition = torch.nn.Conv2d(var_dim_condition, 
                                               var_dim_residual, (1, 1), padding = "same")
        #
        self.layer_conv = MultiScaleDilatedConv(var_dim_residual * 2, 
                                                var_dim_residual * 2, 
                                                var_conv_size_list, 
                                                var_conv_dilation)
        #
        self.layer_output = torch.nn.Conv2d(var_dim_residual, var_dim_residual, (1, 1))

    #
    ##
    def forward(self, 
                var_input, 
                var_step,
                var_condition):
        """
        <parameter>
        : var_input: batch of samples, shape (batch_size, var_dim_residual, var_dim_feature, time)
        : var_step: batch of encoded diffusion steps, shape (batch_size, var_dim_step)
        : var_condition: batch of step-specific conditions, shape (batch_size, var_dim_condition, var_dim_feature, time)
        <return>
        : var_x: output for residual connection
        : var_output: output as an element of final output
        """
        #
        ##  \hat{x}^t
        var_x = var_input
        #
        var_x = self.layer_norm(var_x)
        #
        ## \hat{\bm{t}}
        var_s = self.layer_step(var_step).unsqueeze(-1).unsqueeze(-1)
        #
        var_x = var_x + var_s
        #
        ## \bm{c}_t
        var_c = self.layer_condition(var_condition)[..., :var_x.shape[-1]]
        #
        var_x = torch.cat([var_x, var_c], dim = 1)
        #
        var_x = self.layer_conv(var_x)
        #
        var_gate, var_filter = var_x.split(var_x.size(1) // 2, dim = 1)
        #
        var_x = torch.sigmoid(var_gate) * torch.tanh(var_filter)
        #
        var_output = self.layer_output(var_x)
        #
        var_x = (var_output + var_input) # / (2.0**0.5)
        #
        return var_x, var_output

    #
    ##
    def layer_norm(self,
                   var_input,
                   var_dim = -1):
        """
        <parameter>
        : var_input: batch of samples
        <return>
        : var_output: batch of normalized samples
        """
        #
        ##
        var_mean = torch.mean(var_input, var_dim, keepdim = True)
        var_std = torch.std(var_input, var_dim, keepdim = True)
        #
        var_output = (var_input - var_mean) / (var_std + 1e-12)
        #
        return var_output










## ------------------------------------------------------------------------------------------ ##
## ----------------------------------------- ACDM ------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
class ACDM(torch.nn.Module):
    """
    : module of Adaptive Conditional Diffusion Model (ACDM)
    """
    #
    ##
    def __init__(self,
                 var_num_residual,
                 var_dim_residual,
                 var_dim_feature,
                 var_dim_step,
                 var_dim_condition,
                 var_max_step,
                 var_conv_size_list,
                 var_conv_dilation_cycle):
        """
        <parameter>
        : var_num_residual: number of residual blocks
        : var_dim_residual: dimension of residual block for residual connection
        : var_dim_feature: dimension (CSI channels) of input tensors
        : var_dim_step: dimension of encoded diffusion steps
        : var_dim_condition: dimension of input condition (spectrogram)
        : var_max_step: total number of diffusion steps
        : var_conv_size_list: list of convolutional kernel sizes
        : var_conv_dilation_cycle: cycle of dilation intervals
        """
        #
        super(ACDM, self).__init__()
        #
        ## input projection
        self.layer_input = torch.nn.Conv2d(1, var_dim_residual, (5, 5), padding = "same")
        torch.nn.init.kaiming_normal_(self.layer_input.weight)
        #
        ## step embedding
        self.layer_step = StepEmbedding(var_dim_step, var_max_step)
        #
        ## adaptive conditioner
        ## 1. var_dim_condition = var_nfft / 2 +1
        ## 2. var_scale = var_nfft / 4 = (var_dim_condition - 1) / 2
        self.layer_adaptive_conditioner = AdaptiveConditioner(var_dim_feature = var_dim_feature, 
                                                              var_scale = (var_dim_condition - 1) // 2, 
                                                              var_dim_condition = var_dim_condition, 
                                                              var_dim_step = var_dim_step)
        #
        ## residual blocks
        #
        var_cycle = len(var_conv_dilation_cycle)
        #
        layer_residual_list = [ResidualBlock(var_dim_residual, 
                                             var_dim_step, 
                                             var_dim_condition, 
                                             var_conv_size_list, 
                                             var_conv_dilation = 2 ** (var_conv_dilation_cycle[var_i%var_cycle])) 
                               for var_i in range(var_num_residual)]
        #
        self.layer_residual_list = torch.nn.ModuleList(layer_residual_list)
        #
        ## summary of residuals
        self.layer_summary = torch.nn.Conv2d(var_dim_residual, var_dim_residual//2, (1, 1), padding = "same")
        torch.nn.init.kaiming_normal_(self.layer_summary.weight)
        #
        ## output
        self.layer_output = torch.nn.Conv2d(var_dim_residual//2, 1, (1, 1))
        torch.nn.init.zeros_(self.layer_output.weight)
    
    #
    ##
    def forward(self, 
                var_input, 
                var_step,
                var_condition):
        """
        <parameter>
        : var_input: shape (batch_size, var_dim_feature, time)
        : var_step: shape (batch_size, )
        : var_condition: shape (batch_size, var_dim_feature, var_dim_condition, time_short)
        <return>
        : var_output: shape (batch_size, var_dim_feature, time)
        """
        #
        ## input x^t
        var_x = var_input
        #
        var_x = var_x.unsqueeze(1)
        #
        var_x = self.layer_input(var_x)
        #
        var_x = torch.relu(var_x)
        #
        ## step embedding
        var_s = self.layer_step(var_step)
        #
        ## step-specific conditions
        var_c = self.layer_adaptive_conditioner(var_condition, var_s)
        #
        ## stack of residual blocks
        var_output_sum = None
        #
        for layer_residual in self.layer_residual_list:
            #
            var_x, var_output = layer_residual(var_x, var_s, var_c)
            #
            var_output_sum = var_output if var_output_sum is None else var_output_sum + var_output
        #
        ##
        var_output_sum /= len(self.layer_residual_list) #** 0.5
        #
        ## output layers
        var_output = self.layer_summary(var_output_sum)
        var_output = torch.relu(var_output)
        #
        var_output = self.layer_output(var_output)
        #
        var_output = var_output.squeeze(1)
        #
        return var_output










## ------------------------------------------------------------------------------------------ ##
## ----------------------------------------- Test ------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
def test_data():
    #
    ##
    data_x = np.load(preset["path"]["data"])
    # 
    data_sample_x = np.swapaxes(data_x[0:16], -1, -2)
    #
    data_sample_x = torch.tensor(data_sample_x)
    #
    print(data_sample_x.shape)
    #
    return data_sample_x

#
##
def test_step_embedding():
    #
    ##
    layer_step_embedding = StepEmbedding(var_dim_step = 128, var_max_step = 100)
    #
    print(layer_step_embedding(1).shape)

#
##
def test_adaptive_conditioner():
    #
    ##
    data_sample_x = test_data()
    #
    ##
    var_dim_step = 128
    var_max_step = 100
    #
    ## step embedding
    layer_step = StepEmbedding(var_dim_step, var_max_step)
    #
    ## adaptive conditioner
    layer_adaptive_conditioner = AdaptiveConditioner(var_dim_feature = data_sample_x.shape[1], 
                                                     var_scale = 64, 
                                                     var_dim_condition = 129, 
                                                     var_dim_step = var_dim_step)
    #
    data_spectrogram = torchaudio.transforms.Spectrogram(**preset["spectrogram"])(data_sample_x)
    #
    print("0", data_spectrogram.shape)
    #
    var_output = layer_adaptive_conditioner(data_spectrogram, layer_step(1))
    #
    print("1", var_output.shape)

#
##
def test_acdm():
    #
    ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    ##
    data_sample_x = test_data()
    #
    print("data_sample_x", data_sample_x.shape)
    #
    ##
    data_spectrogram = torchaudio.transforms.Spectrogram(**preset["spectrogram"])(data_sample_x)
    #
    print("data_spectrogram", data_spectrogram.shape)
    #
    ##
    model_acdm = ACDM(var_dim_condition = 129,
                      **preset["acdm"]).to(device)
    #
    var_output = model_acdm(data_sample_x.to(device), 
                            torch.randint(0, 100, [16]), 
                            data_spectrogram.to(device))
    #
    print("var_output", var_output.shape)










## ------------------------------------------------------------------------------------------ ##
## ----------------------------------------- Main ------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
if __name__ == "__main__":
    #
    ##
    # test_data()
    #
    # test_step_embedding()
    #
    # test_adaptive_conditioner()
    #
    test_acdm()







