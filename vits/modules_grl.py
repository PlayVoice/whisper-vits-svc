# Adapted from https://github.com/ubisoft/ubisoft-laforge-daft-exprt Apache License Version 2.0
# Unsupervised Domain Adaptation by Backpropagation

import torch
import torch.nn as nn

from torch.autograd import Function
from torch.nn.utils import weight_norm


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


class SpeakerClassifier(nn.Module):

    def __init__(self, embed_dim, spk_dim):
        super(SpeakerClassifier, self).__init__()
        self.classifier = nn.Sequential(
            GradientReversal(lambda_reversal=1),
            weight_norm(nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(embed_dim, spk_dim, kernel_size=5, padding=2))
        )

    def forward(self, x):
        ''' Forward function of Speaker Classifier:
            x = (B, embed_dim, len)
        '''
        # pass through classifier
        outputs = self.classifier(x)  # (B, nb_speakers)
        outputs = torch.mean(outputs, dim=-1)
        return outputs
