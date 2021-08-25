import torch
import torch.nn as nn
import functools
from utils.custom_modules import Conv2dEncoder, Conv2dDecoder
from utils.training_process import TrainingProcess
from typing import Union, List


class CRUSE(nn.Module):
    """ Convolutional Recurrent U-Net for Speech Enhancement (CRUSE)

    Implementation of CRUSE as described in:
    https://arxiv.org/pdf/2101.09249.pdf

    The CRUSE model consist of a symmetric convolutional encoder/decoder and a
    recurrent layer between the encoder and decoder. The input features are the
    log power spectrum of a noisy speech signal and the output features
    correspond to the mask that will be multiplied by the complex spectrum to
    generate a clean speech signal.

    Args:
        sample_rate (int): Assumed sample rate of the input tensor.
        window_size (int): Assumed window size of the input tensor.
        hop_size (int): Assumed hop size of the input tensor.
        sample_duration (float): Assumed duration in seconds of the 
            input tensor. 
        batch_size (int): Batch size used to feed data into the model.
        kernel_size (tuple): Kernel size of encoder and decoder filters.
        stride (tuple): Stride usee in encoder and decoder filter convolutions.
        chl_seq (list): List of sizes of the encoder. The decoder will use the
            reverse order.
        bottleneck_size (int): Number of GRU between encoder and decoder. 
            The input features are flattened and equally split between all 
            RNN layers.
        eps (float): Machine epsilon used for internal computations of the 
            model.
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 window_size: int = 320,
                 hop_size: int = 160,
                 sample_duration: float = 10.0,
                 batch_size: int = 10,
                 kernel_size: tuple = (2, 3),
                 stride: tuple = (1, 2),
                 chl_seq: list = [1, 16, 32, 64, 64],
                 bottleneck_size: int = 1,
                 encoder_bias: bool = True,
                 decoder_bias: bool = True,
                 bottleneck_bias: bool = True,
                 eps: float = 1e-10):
        super().__init__()

        # audio params
        self.sample_rate = sample_rate
        self.window_size = window_size
        self._window = torch.hann_window(self.window_size)
        self.hop_size = hop_size
        self.sample_duration = sample_duration
        self.eps = eps

        # network params
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.chl_seq = chl_seq
        self.bottleneck_size = bottleneck_size
        self.encoder_bias = encoder_bias
        self.decoder_bias = decoder_bias
        self.bottleneck_bias = bottleneck_bias

        # encoder
        self.encoder = Conv2dEncoder(self.chl_seq,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     return_skips=True)

        # bottleneck
        self.bottleneck = nn.ModuleList([
            nn.GRU(self.bottleneck_element_dims[1], 
                   self.bottleneck_element_dims[1])
            for idx in range(self.bottleneck_size)
            ])

        # decoder
        self.decoder = Conv2dDecoder(self.chl_seq[::-1], # reverse order
                                     kernel_size=self.kernel_size,
                                     stride=self.stride)

        self.init_weights_and_biases()

        # probe for testing (created here to assure all tensors are on
        # the same device)
        self._probe = torch.randn([self.batch_size, self.sample_len],
                                   dtype=torch.float32)

    def stft(self, x: torch.tensor, normalize: bool = False,
             complex_output: bool = False):
        """ Return the short-time Fourier transform of a tensor.

        Args:
            x (torch.tensor): Input tensor.
            normalize (bool): If True, the stft values are normalized.
            complex_output (bool): If True, the output is return in complex
                form.
        """
        window = self._window.to(x.device)

        stft = torch.stft(x, onesided=True, center=True,
                          n_fft=self.window_size, hop_length=self.hop_size,
                          normalized=normalize, window=window,
                          return_complex=True)

        if complex_output:
            return stft
        else:
            return torch.abs(stft), torch.angle(stft)

    def istft(self, x_mag: torch.tensor, x_phase: torch.tensor):
        """ Return the inverse short-time fourier transform of a tensor.

        Args:
            x_mag (torch.tensor): Input tensor containing the magnitude values.
            x_phase (torch.tensor): Input tensor containing the phase values.

        Returns:
            istft (torch.tensor): Time domain signal based on x_mag and x_phase.
        """
        x_real = x_mag * torch.cos(x_phase)
        x_imag = x_mag * torch.sin(x_phase)
        x_complex = torch.compelx(x_real, x_imag)

        istft = torch.istft(x_complex,
                            onesided=True,
                            center=True,
                            n_fft=self.window_size,
                            hop_length=self.hop_size,
                            normalized=False,
                            window=self._window)

        return istft

    @property
    @functools.lru_cache()
    @torch.no_grad()
    def bottleneck_element_dims(self):
        """ Calculate the dimensions of a single bottleneck element.

        If the bottleneck is comprised of a single element, the input dimension
        corresponds to the flattened features along the channel axis, otherwise
        these features are equally split across elements.

        IMPORTANT: The encoder's output is assumed to be a 4D tensor in the
            order (batch_size, channels, time_steps, features).

        Returns:
            bottleneck_element_dims (list): Returns a list with the dimensions
                to be used on each bottleneck element.

        Raises:
            RuntimeError: If the flattened features are not divisible by the
                number of bottleneck elements.
        """

        # initial piece of audio
        x = torch.randn([self.batch_size, self.sample_len],
                         dtype=torch.float32).to(next(self.parameters()).device)

        # extracts magnitude and phase
        x_mag, x_phase = self.stft(x, complex_output=False)

        # get pow log spectrum
        x_mag_pow_log = torch.log10(x_mag ** 2 + self.eps)

        # ends with shape (batch_size, channels, time_steps, bins)
        x_mag_pow_log = x_mag_pow_log.unsqueeze(1).permute(0, 1, 3, 2)

        # encoder processing
        encoder_output_dims = self.encoder.get_output_dims(x_mag_pow_log)
        
        # flattened features along the channel axis
        flattened_features = encoder_output_dims[1] * encoder_output_dims[3]

        if self.bottleneck_size == 1:
            return [
                    encoder_output_dims[0],
                    flattened_features,
                    encoder_output_dims[2]
                    ]
        else:
            if flattened_features % self.bottleneck_size != 0:
                raise RuntimeError("Unable to split flattened features "
                        f"({flattened_features}) equally into the chosen "
                        f"number of bottleneck elements ({self.bottleneck_size})"
                        )

            return [
                    encoder_output_dims[0],
                    flattened_features // self.bottleneck_size,
                    encoder_output_dims[2]
                    ]

    @property
    def sample_len(self):
        """ Returns the amount of samples on each input audio chunk. 

        Returns:
            sample_len (int): Audio chunk length in samples
        """
        return int(self.sample_duration * self.sample_rate)

    def init_weights_and_biases(self):
        """ Initialize weights and biases based on layer types. """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                self.init_conv2d_(module)
            
            if isinstance(module, nn.ConvTranspose2d):
                self.init_convtranspose2d_(module)

            if isinstance(module, nn.GRU):
                self.init_gru_(module)

    def init_conv2d_(self, conv2d_l):
        torch.nn.init.xavier_uniform_(conv2d_l.weight)

        if conv2d_l.bias is not None:
            torch.nn.init.zeros_(conv2d_l.bias)

    def init_convtranspose2d_(self, convtranspose2d_l):
        self.init_conv2d_(convtranspose2d_l)

    def init_gru_(self, gru_l):
        torch.nn.init.xavier_uniform_(gru_l.weight_ih_l0)
        torch.nn.init.xavier_uniform_(gru_l.weight_hh_l0)

        if gru_l.bias is not None:
            torch.nn.init.zeros_(gru_l.bias_ih_l0)
            torch.nn.init.zeros_(gru_l.bias_hh_l0)

    @torch.no_grad()
    def predict(self, x: torch.tensor):
        x_complex = self.stft(x, complex_output=True)
        x_log_pow_spec = torch.log10(torch.abs(x_complex) ** 2 + self.eps)
        pred_mask = self(x_log_pow_spec)
        pred_output_complex = (
                pred_mask.squeeze(1).permute(0, 2, 1) * x_complex)

        pred_x = torch.istft(pred_output_complex,
                             onesided=True,
                             n_fft=self.window_size,
                             center=True,
                             hop_length=self.hop_size,
                             normalized=False,
                             window=self._window.to(x.device))

        pred_x = pred_x.cpu().numpy().reshape(-1)

        return pred_x

    def forward(self, x):
        # adjust input tensor dimensions
        x = x.unsqueeze(1).permute(0, 1, 3, 2)

        # encoder and skip connections
        x, skips = self.encoder(x)

        # flattened features along the channel axes
        flat_x = x.permute(2, 0, 1, 3).flatten(start_dim=2)

        # split flattened dimension in bottleneck_size bottleneck elements
        bneck_input = torch.split(flat_x,
                                  self.bottleneck_element_dims[1], dim=-1)

        # store bottleneck outputs of each GRU
        bneck_output = []

        for idx, bneck_input_split in enumerate(bneck_input):
            bneck_partial, _ = self.bottleneck[idx](bneck_input_split)
            bneck_output.append(bneck_partial)

        # flattened concatenated output to be fed to the decoder
        concat_bneck_output = torch.cat(tuple(bneck_output), dim=-1).view_as(x)

        # decoder
        decoder_output = self.decoder(concat_bneck_output, skips)

        return decoder_output


class CRUSETrainingProcess(TrainingProcess):
    """ Callbacks executed in different phases of the training process of a
        CRUSE instance. For further information check the TrainingProcess
        parent class. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_step(self, batch_idx, batch):
        ((x_complex, x_log_pow_spec),
         (y_complex, y_log_pow_spec)) = batch

        x_complex = x_complex.to(self.device)
        x_log_pow_spec = x_log_pow_spec.to(self.device)
        y_complex = y_complex.to(self.device)
        y_log_pow_spec = y_log_pow_spec.to(self.device)

        y_pred_mask = self.model(x_log_pow_spec)
        loss = self.criterion(y_pred_mask, x_complex, y_complex)
        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_norm_clipping)

        self.optimizer.step()
        self.running_dict.set_value("train_loss", loss.item())

    def on_val_step(self, batch_idx, batch):
        ((x_complex, x_log_pow_spec),
         (y_complex, y_log_pow_spec)) = batch

        x_complex = x_complex.to(self.device)
        x_log_pow_spec = x_log_pow_spec.to(self.device)
        y_complex = y_complex.to(self.device)
        y_log_pow_spec = y_log_pow_spec.to(self.device)

        y_pred_mask = self.model(x_log_pow_spec)
        loss = self.criterion(y_pred_mask, x_complex, y_complex)
        self.running_dict.set_value("val_loss", loss.item())

    def on_overfit_train_step(self, batch_idx, batch):
        self.on_train_step(batch_idx, batch)
