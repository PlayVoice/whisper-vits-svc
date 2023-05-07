# Adapted from https://github.com/keonlee9420/Daft-Exprt MIT License
# Adapted from https://github.com/ubisoft/ubisoft-laforge-daft-exprt Apache License Version 2.0

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

    def __init__(self, hparams):
        super(GradientReversal, self).__init__()
        self.lambda_ = hparams.lambda_reversal

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


class SpeakerRClassifier(nn.Module):
    ''' Speaker Classifier Module:
        - 3x Linear Layers with ReLU
    '''

    def __init__(self, hparams):
        super(SpeakerRClassifier, self).__init__()
        nb_speakers = hparams.n_speakers - 1
        embed_dim = hparams.prosody_encoder['hidden_embed_dim']
        self.classifier = nn.Sequential(
            GradientReversal(hparams),
            LinearNorm(embed_dim, embed_dim, w_init_gain='relu'),
            nn.ReLU(),
            LinearNorm(embed_dim, embed_dim, w_init_gain='relu'),
            nn.ReLU(),
            LinearNorm(embed_dim, nb_speakers, w_init_gain='linear')
        )

    def forward(self, x):
        ''' Forward function of Speaker Classifier:
            x = (B, embed_dim)
        '''
        # pass through classifier
        outputs = self.classifier(x)  # (B, nb_speakers)
        return outputs


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__()
        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return RevGrad.apply(input_, self._alpha)


class RevGrad(torch.autograd.Function):
    """
    A gradient reversal layer.
    This layer has no parameters, and simply reverses the gradient in the backward pass.
    See https://www.codetd.com/en/article/11984164, https://github.com/janfreyberg/pytorch-revgrad
    """
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    , extended to 'TADAM: Task dependent adaptive metric for improved few-shot learning'
    """

    def __init__(self):
        super(FiLM, self).__init__()
        self.s_gamma = nn.Parameter(torch.ones(1,), requires_grad=True)
        self.s_beta = nn.Parameter(torch.ones(1,), requires_grad=True)

    def forward(self, x, gammas, betas):
        """
        x -- [B, T, H]
        gammas -- [B, 1, H]
        betas -- [B, 1, H]
        """
        gammas = self.s_gamma * gammas.expand_as(x)
        betas = self.s_beta * betas.expand_as(x)
        return (gammas + 1.0) * x + betas


class SpeakerClassifier(nn.Module):
    """ Speaker Classifier """

    def __init__(self, model_config):
        super(SpeakerClassifier, self).__init__()
        n_speaker = model_config["n_speaker"]
        input_dim = model_config["prosody_encoder"]["encoder_hidden"]
        self.hidden = model_config["prosody_encoder"]["encoder_hidden"]
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_dim, self.hidden)),
            ('ln1', nn.LayerNorm(self.hidden)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.hidden, self.hidden)),
            ('ln2', nn.LayerNorm(self.hidden)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(self.hidden, n_speaker)),
            ('softmax', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, x):
        # GRL
        rev_x = self.grl(x)
        # Calculate augmentation posterior
        score = self.classifier(rev_x)
        if len(score.size()) > 2:
            score = score.mean(dim=1)
        return score  # [batch, 2]


# -- Speaker Classifier
# speaker_posterior = self.speaker_classifier(prosody_vector.squeeze(1))
# LOSS
# self.nll_loss = nn.NLLLoss()
# adv_loss = self.nll_loss(speaker_posteriors, speaker_targets[序号])
