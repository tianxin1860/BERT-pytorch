import torch.nn as nn
import torch
import types


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        print("Types of LayerNorm's input:{0}".format(type(x)))
        x = x.float()
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        print("x shape:{0}".format(x.shape))
        print("a_2 shape:{0}".format(self.a_2.shape))
        print("b_2 shape:{0}".format(self.b_2.shape))
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
