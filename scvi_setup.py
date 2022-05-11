# torch imports
import torch
from torch import nn as nn
from torch.nn import ModuleList
from torch.nn import functional as F
from torch import logsumexp
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl

# scvi imports
import scvi
from scvi._compat import Literal
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    LayerField,
    CategoricalObsField,
    NumericalObsField,
    CategoricalJointObsField,
    NumericalJointObsField,
)
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, LinearDecoderSCVI, one_hot, FCLayers


# other imports
import numpy as np
import collections
from typing import Callable, Iterable, List, Optional

# settings
scvi.settings.seed = 0
torch.backends.cudnn.benchmark = True