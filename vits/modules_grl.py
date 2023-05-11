# Adapted from https://github.com/ubisoft/ubisoft-laforge-daft-exprt Apache License Version 2.0
# Unsupervised Domain Adaptation by Backpropagation

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    ''' Gradient Reversal Layer
            Y. Ganin, V. Lempitsky,
            "Unsupervised Domain Adaptation by Backpropagation",
            in ICML, 2015.
        Forward pass is the identity function
        In the backward pass, upstream gradients are multiplied by -lambda (i.e. gradient are reversed)
    '''

    def __init__(self, lambda_reversal=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_reversal

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class LinearNorm(nn.Module):
    ''' Linear Norm Module:
        - Linear Layer
    '''

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight,
                                gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        ''' Forward function of Linear Norm
            x = (*, in_dim)
        '''
        x = self.linear_layer(x)  # (*, out_dim)

        return x


class SpeakerClassifier(nn.Module):
    ''' Speaker Classifier Module:
        - 3x Linear Layers with ReLU
    '''

    def __init__(self, embed_dim, spk_dim):
        super(SpeakerClassifier, self).__init__()
        self.classifier = nn.Sequential(
            GradientReversal(lambda_reversal=1),
            LinearNorm(embed_dim, embed_dim, w_init_gain='relu'),
            nn.ReLU(),
            LinearNorm(embed_dim, embed_dim, w_init_gain='relu'),
            nn.ReLU(),
            LinearNorm(embed_dim, spk_dim, w_init_gain='linear')
        )

    def forward(self, x):
        ''' Forward function of Speaker Classifier:
            x = (B, embed_dim)
        '''
        # pass through classifier
        x_ave = torch.mean(x, dim=-1)
        outputs = self.classifier(x_ave)  # (B, nb_speakers)
        return outputs
