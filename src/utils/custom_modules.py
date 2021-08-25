import torch
import torch.nn as nn
from typing import List, Union


# TODO: allow additional kwargs
class CausalConv1d(nn.Conv1d):
    """ Causal Convolutional 1D layer.

    A simple nn.Conv1d with causal padding.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self._padding = (kernel_size - 1) * dilation

        super().__init__(
                 in_channels,
                 out_channels,
                 kernel_size=kernel_size,
                 stride=stride,
                 padding=self._padding,
                 dilation=dilation,
                 groups=groups,
                 bias=bias
                )

    def forward(self, x):
        result = super().forward(x)

        if self._padding != 0:
            return result[:, :, : - self._padding]

        return result


class Conv2dEncoder(nn.Module):
    """ 2D convolutional encoder.

    Args:
        chl_seq (list): Input/Output channel sequence used in every Conv2d of 
            the encoder. The number of conv2d layers are equal to:
            len(chl_seq) - 1.
        kernel_size (tuple): Kernel size used on each Conv2d layer.
        stride (stride): Stride used on each Conv2d layer.
        activation (nn.Module): Activation function computed after every Conv2d.
        return_skips (bool): If True, forward() returns the resulting tensors
            along with the resulting skip connection after every Conv2d.
    """
    def __init__(self,
                 chl_seq: list = [1, 16, 32, 64, 64],
                 kernel_size: tuple = (2, 3),
                 stride: tuple = (1, 2),
                 activation_seq: List[nn.Module] = [
                     nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU],
                 return_skips: bool = True):
        super().__init__()

        # hparams
        self.chl_seq = chl_seq
        self.kernel_size = kernel_size
        self.stride = stride
        self.return_skips = return_skips

        # encoder
        self.encoder_l = nn.ModuleList()

        for idx in range(len(chl_seq) - 1):
            self.encoder_l.add_module(f"conv2d_l{idx}",
                                      nn.Conv2d(in_channels=chl_seq[idx],
                                                out_channels=chl_seq[idx + 1],
                                                kernel_size=self.kernel_size,
                                                stride=self.stride))

            self.encoder_l.add_module(f"activation_{idx}", 
                                      activation_seq[idx]())

    @torch.no_grad()
    def get_output_dims(self, x):
        """ Return the tensor output dimensions by probing with a tensor x.

        Args:
            x (torch.tensor): Probe tensor.

        Returns:
            (torch.Size): Output shape of the encoder.
        """
        return self(x)[0].shape if self.return_skips else self(x).shape

    def forward(self, x):
        if self.return_skips:
            skips = []

        for idx in range(len(self.encoder_l) // 2):
            x = self.encoder_l._modules[f"conv2d_l{idx}"](x)
            x = self.encoder_l._modules[f"activation_{idx}"](x)

            if self.return_skips:
                skips.append(x)
        
        return (x, skips) if self.return_skips else x


class Conv2dDecoder(nn.Module):
    """ 2D convolutional decoder. """
    def __init__(self,
                 chl_seq: list = [64, 64, 32, 16, 1],
                 # IMPORTANT: len(chl_seq) = len(padding_seq) + 1
                 output_padding_seq: Union[List[int], List[tuple]] = [
                     0, 0, (0, 1), 0],
                 kernel_size: tuple = (2, 3),
                 stride: tuple = (1, 2),
                 activation_seq: List[nn.Module] = [
                     nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU, nn.Sigmoid
                     ],
                 last_activation: nn.Module = nn.Sigmoid):
        super().__init__()

        # hparams
        self.chl_seq = chl_seq
        self.output_padding_seq = output_padding_seq
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation_seq = activation_seq

        # decoder
        self.decoder_l = nn.ModuleList()

        for idx in range(len(chl_seq) - 1):
            self.decoder_l.add_module(f"convtranspose2d_l{idx}",
                    nn.ConvTranspose2d(in_channels=chl_seq[idx],
                                       out_channels=chl_seq[idx + 1],
                                       kernel_size=self.kernel_size,
                                       stride=self.stride,
                                       output_padding=output_padding_seq[idx]))

            self.decoder_l.add_module(f"activation_{idx}",
                                      activation_seq[idx]())

    def forward(self, x, skips):
        for idx in range(len(self.decoder_l) // 2):

            if skips is not None:
                x += skips[-(1 + idx)]  # reverse order

            x = self.decoder_l._modules[f"convtranspose2d_l{idx}"](x)
            x = self.decoder_l._modules[f"activation_{idx}"](x)

        return x


class DTLNSeparationCore(nn.Module):
    """ Dual-Signal Transformation Long Shot-Term Memory Network separation core
    as described in:

    https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2631.pdf

    The original network uses two separation cores with a stack of LSTM
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 rnn_type: nn.Module = nn.LSTM,
                 rnn_stack_size: int = 2,
                 rnn_bidirectional: bool = False,
                 dropout_rate: float = 0.25,
                 activation: nn.Module = nn.Sigmoid):
        super().__init__()

        # network params
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn_type = rnn_type
        self.rnn_stack_size = rnn_stack_size
        self.rnn_bidirectional = rnn_bidirectional
        self.dropout_rate = dropout_rate
        self.activation = activation

        # network modules
        self.separation_core = nn.ModuleDict({
            "rnn_stack": self.rnn_type(self.input_size, self.hidden_size,
                                       dropout=self.dropout_rate,
                                       num_layers=self.rnn_stack_size,
                                       bidirectional=self.rnn_bidirectional),
            "fcl": nn.Linear(self.hidden_size * (
                                    2 if self.rnn_bidirectional else 1), 
                             self.output_size),
            "activation": self.activation()
            })

    def forward(self, x):
        x, _ = self.separation_core["rnn_stack"](x)
        x = x.permute(1, 0, 2)
        x = self.separation_core["fcl"](x)
        x = self.separation_core["activation"](x)
        return x
