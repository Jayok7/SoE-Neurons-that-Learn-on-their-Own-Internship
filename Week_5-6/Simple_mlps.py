"""
This week has establishing the foundations as one of the key goals in mind
For now, one of the objectives to that end is to reduce extra noise such as extra metrics
Main metrics to be looked at will be CE loss, recon_loss and model accuracy
Interleaved scheduling at p=0.5 will be used alongside combined loss and j
Archs will be typical NN, ANN and a normal NAN
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class PlainNNlayer(nn.module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = nn.linear(in_dim, out_dim)

        self.decoder_bias = nn.parameters(torch.zeros(out_dim, in_dim, requires_grad=False))
        self.decoder_weights = nn.parameters(torch.empty(out_dim, in_dim, requires_grad=False))
        

class AutoencoderNN:
    def __init__(nn.module):
        super().__init__():

class 

ARCHS = {
    "NN" : PlainNNlayer,
    "ANN": AutoencoderNN,
    "NAN": 
}