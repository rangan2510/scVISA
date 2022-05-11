import torch
from torch import nn as nn
from typing import Callable, Iterable, Optional
from scvi.nn import DecoderSCVI, Encoder, LinearDecoderSCVI, one_hot, FCLayers

class Attention(nn.Module):
    def __init__(self, in_feat,out_feat):
        super().__init__()             
        self.Q = nn.Linear(in_feat,out_feat) # Query
        self.K = nn.Linear(in_feat,out_feat) # Key
        self.V = nn.Linear(in_feat,out_feat) # Value
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        d = K.shape[0] # dimension of key vector
        QK_d = (Q @ K.T)/(d)**0.5
        prob = self.softmax(QK_d)
        attention = prob @ V
        return attention

