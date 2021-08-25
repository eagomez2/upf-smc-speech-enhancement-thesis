import torch
import torch.nn as nn
from utils.custom_modules import CausalConv1d, DTLNSeparationCore
from utils.training_process import TrainingProcess


class DTLN(nn.Module):
    """ Dual-Signal Transformation Long Short-Term Memory Network.

    Based on the work presented by N. Westhausen et. al for the
    DNS INTERSPEECH 2020:

    https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2631.pdf

    The original implementation was coded in tensorflow 2.x. The current
    implementation is based on it with some changes to attempt to improve
    the existing model for noise suppression tasks on speech signals.

    Args:
        sample_rate (int): Sample rate of the input files.
        window_size (int): Window (and FFT) size used to perform the STFT over 
            the input files.
        hop_size (int): Hop size used to perform the STFT over the input files.
        sample_duration (int): Duration in seconds of each input file.
        hidden_size (int): Hidden size of the RNN stack.
        encoder_size (int): Channel output size of Conv1D layers used in the
            network (see architecture for the details).
        rnn_type (nn.Module): Type of RNN layer used in the stack.
        rnn_stack_size (int): Size of the RNN stack.
        rnn_bidirectional (bool): If True, makes the RNN layer bidirectional.
        dropout_rate (float): Dropout rate used between RNN stacks.
        batch_size (int): Size of a single batch.
        eps (float): Machine epsilon.
    """
    def __init__(self,
                 sample_rate: int = 16000,
                 window_size: int = 512,
                 hop_size: int = 128,
                 sample_duration: float = 10.0,
                 hidden_size: int = 128,
                 encoder_size: int = 256,
                 rnn_type: nn.Module = nn.LSTM,
                 rnn_stack_size: int = 2,
                 rnn_bidirectional: bool = False,
                 dropout_rate: float = 0.25,
                 batch_size: int = 32,
                 eps: float = 1e-10):
        super().__init__()

        # audio parmeters
        self.sample_rate = sample_rate
        self.window_size = window_size
        self._window = torch.hann_window(self.window_size)
        self.hop_size = hop_size
        self.sample_duration = sample_duration
        self.eps = eps

        # network params
        self.hidden_size = hidden_size
        self.encoder_size = encoder_size
        self.dropout_rate = dropout_rate
        self.rnn_type = rnn_type
        self.rnn_stack_size = rnn_stack_size
        self.rnn_bidirectional = rnn_bidirectional
        self.batch_size = batch_size

        # network modules 
        self.layer_norm_0 = nn.LayerNorm(self.window_size // 2 + 1, 
                                          eps=self.eps)
        # first separation core
        self.separation_core_0 = DTLNSeparationCore(
                    self.window_size // 2 + 1,
                    self.hidden_size,
                    self.window_size // 2 + 1,
                    rnn_type=self.rnn_type, 
                    rnn_stack_size=self.rnn_stack_size,
                    rnn_bidirectional=self.rnn_bidirectional,
                    activation=nn.Sigmoid
                    )

        # modules between separation cores
        self.conv1d_1 = nn.Conv1d(self.window_size, self.encoder_size,
                                  kernel_size=1, stride=1, bias=False)
        self.layer_norm_1 = nn.LayerNorm(self.encoder_size, eps=self.eps)

        # second separation core
        self.separation_core_1 = DTLNSeparationCore(
                    self.encoder_size,
                    self.hidden_size,
                    self.encoder_size,
                    rnn_type=self.rnn_type, rnn_stack_size=self.rnn_stack_size,
                    rnn_bidirectional=self.rnn_bidirectional,
                    activation=nn.Sigmoid
                    )

        # additional tail block
        self.causal_conv1d_1 = CausalConv1d(self.encoder_size,
                                            self.window_size,
                                            kernel_size=1, bias=False)

        self.overlap_and_add = nn.Fold((1, self.sample_len),
                                       kernel_size=(1, self.window_size),
                                       stride=(1, self.hop_size))

        # init weights and biases
        self.init_weights_and_biases()

    def init_weights_and_biases(self):
        """ Initialize weights and biases based on each module type. """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                self.init_linear_(module)

            if isinstance(module, nn.LayerNorm):
                self.init_layer_norm_(module)

            if isinstance(module, nn.GRU):
                self.init_gru_(module)

            if isinstance(module, nn.LSTM):
                self.init_lstm_(module)
    
    def init_lstm_(self, lstm_l):
        torch.nn.init.xavier_uniform_(lstm_l.weight_ih_l0)
        torch.nn.init.xavier_uniform_(lstm_l.weight_hh_l0)

        if lstm_l.bias is not None:
            torch.nn.init.zeros_(lstm_l.bias_ih_l0)
            torch.nn.init.zeros_(lstm_l.bias_hh_l0)
    
    def init_gru_(self, gru_l):
        torch.nn.init.xavier_uniform_(gru_l.weight_ih_l0)
        torch.nn.init.xavier_uniform_(gru_l.weight_hh_l0)

        if gru_l.bias is not None:
            torch.nn.init.zeros_(gru_l.bias_ih_l0)
            torch.nn.init.zeros_(gru_l.bias_hh_l0)

    def init_linear_(self, linear_l):
        torch.nn.init.xavier_uniform_(linear_l.weight)

        if linear_l.bias is not None:
            torch.nn.init.zeros_(linear_l.bias)

    def init_layer_norm_(self, layer_norm):
        torch.nn.init.ones_(layer_norm.weight)

        if layer_norm.bias is not None:
            torch.nn.init.zeros_(layer_norm.bias)

    def stft(self, x: torch.tensor, normalize: bool = False,
             complex_output: bool = False):
        """ Return the short-time fourier transform of a tensor. 

        Args:
            x (torch.tensor): Input tensor.
            normalize (bool): If True, the stft values are normalized.
            complex_output (bool): If True, the output is return in complex
                form.

        Returns:
            stft (torch.tensor): STFT of the input tensor.
        """
        window = self._window.to(x.device)

        stft = torch.stft(x, onesided=True, center=False,
                n_fft=self.window_size, hop_length=self.hop_size,
                normalized=normalize, window=window,
                return_complex=True)

        if complex_output:
            return stft
        else:
            return torch.abs(stft), torch.angle(stft)

    def ifft(self, x_mag: torch.tensor, x_phase: torch.tensor):
        """ Return the inverse fourier transform of a pair of magnitude
        and phase tensors.

        IMPORTANT: This function assumes the reconstruction is done using only
        first half of the input features.

        Args:
            x_mag (torch.tensor): Magnitude input tensor.
            x_phase (torch.tensor): Phase input tensor.

        Returns:
            ifft (torch.tensor): Inverse fourier transform of the input tensor.
        """
        x_real = x_mag * torch.cos(x_phase)
        x_imag = x_mag * torch.sin(x_phase)
        x_complex = torch.complex(x_real, x_imag)
        ifft = torch.fft.irfft(x_complex, dim=-1)
        return ifft

    @property
    def sample_len(self):
        """ Returns the amount of samples on each input audio chunk. 
        Returns:
            sample_len (int): Audio chunk length in samples
        """
        return int(self.sample_duration * self.sample_rate)

    @torch.no_grad()
    def predict(self, x: torch.tensor):
        """ Takes a input frame of noisy speech in the time domain and
        produces a clean speech output frame.

        Args:
            x (torch.tensor): Input noisy speech frame.

        Returns:
            (torch.tensor): Output predicted clean speech frame. 
        """
        return self(x).cpu().numpy().reshape(-1)

    def forward(self, x):
        # get magnitude and phase in a (batch_size, bins, frames) tensor
        x_mag, x_phase = self.stft(x)

        # obtain log spectrum
        x_mag_log = torch.log10(x_mag + self.eps)

        # (batch_size, bins, frames) -> (batch_size, frames, bins)
        x_mag_log = x_mag_log.permute(0, 2, 1)

        # norm with learnable weights (batch_size, frames, bins)
        x_mag_log = self.layer_norm_0(x_mag_log)

        # (batch_size, frames, bins) -> (frames, batch_size, bins)
        x_mag_log = x_mag_log.permute(1, 0, 2)

        # first separation core -> (batch_size, frames, bins)
        x_mask_0 = self.separation_core_0(x_mag_log)
        
        # (batch_size, bins, frames)
        x_mag_embedding = x_mag.permute(0, 2, 1) * x_mask_0

        # back to "time" domain
        x = self.ifft(x_mag_embedding, x_phase.permute(0, 2, 1))

        # (batch_size, frames, samples) -> (batch_size, samples, frames)
        x = x.permute(0, 2, 1)

        # mask to be multiplied by the output of the second separation core
        x_skip_1 = self.conv1d_1(x)

        # (batch_size, features, frames) -> (batch_size, frames, features)
        x = x_skip_1.permute(0, 2, 1)

        # instant layer norm
        x = self.layer_norm_1(x)

        # (batch_size, frames, features) -> (frames, batch_size, features)
        x = x.permute(1, 0, 2)

        # second separation core -> (batch_size, frames, features)
        x_mask_1 = self.separation_core_1(x)

        # (batch_size, features, frames)
        x = x_mask_1.permute(0, 2, 1) * x_skip_1

        # causal conv before overlap and add
        x = self.causal_conv1d_1(x)

        # reconstruction to time domain
        x = self.overlap_and_add(x)

        # reshape output vector to (chl, samples)
        x = torch.reshape(x, (self.batch_size, -1))

        return x
        

class DTLNTrainingProcess(TrainingProcess):
    """ Callbacks executed in different phases of the training process of a
        DTLN instance. For further information check the TrainingProcess
        parent class. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_step(self, batch_idx, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_norm_clipping)

        self.optimizer.step()
        self.running_dict.set_value("train_loss", loss.item())

    def on_val_step(self, batch_idx, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)

        self.running_dict.set_value("val_loss", loss.item())

    def on_overfit_train_step(self, batch_idx, batch):
        self.on_train_step(batch_idx, batch)
