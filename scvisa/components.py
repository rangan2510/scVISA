import torch
from torch import nn as nn
from typing import Callable, Iterable, Optional
from scvi.nn import DecoderSCVI, Encoder, LinearDecoderSCVI, one_hot, FCLayers
try:
    from utils import reparameterize_gaussian, identity
    from layers import Attention
except:
    from scvisa.utils import reparameterize_gaussian, identity
    from scvisa.layers import Attention

# Encoder
class RSAEncoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.att1 = Attention(n_hidden,n_hidden)
        self.att2 = Attention(n_hidden,n_hidden)
        self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity
        self.var_activation = torch.exp if var_activation is None else var_activation
        print("Initialized Self-Attention Encoder")
        self.bn = nn.BatchNorm1d(n_hidden)

    def forward(self, x: torch.Tensor, *cat_list: int):
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        qa = self.att1(q)
        q =  torch.sigmoid(qa +float(self.alpha)*q)
        # q = self.att2(q)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        latent = self.z_transformation(reparameterize_gaussian(q_m, q_v))
        return q_m, q_v, latent