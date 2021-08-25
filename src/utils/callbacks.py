import os
import torch
import glob
import warnings
import soundfile
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from .training_process import TrainingProcessCallback
from collections.abc import Callable
from typing import Union

# ignore matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ScheduledCheckpointCallback(TrainingProcessCallback):
    """ Callback that allows saving a checkpoint every a fixed
    number of epochs during a network training process.

    Args:
        epoch_interval (int): Number of epoch between saved checkpoints.
        checkpoint_prefix (str): Prefix prepended to each checkpoint.
        start_epoch (int): Epoch where the first checkpoint is saved.
        verbose (bool): If True, verbose mode is enabled.
    """
    def __init__(self,
            epoch_interval: int = 25,
            checkpoint_prefix: str = "scheduled_",
            start_epoch: int = 0,
            verbose: bool = False):
        super().__init__()
        self.epoch_interval = epoch_interval
        self.checkpoint_prefix = checkpoint_prefix
        self.start_epoch = start_epoch
        self._ts_fmt = "%Y-%m-%d %H:%M:%S"
        self.verbose = verbose

    def on_val_epoch_end(self, training_process):
        current_epoch = training_process.current_epoch

        # epochs are 0-indexed
        if (self.start_epoch > 0 and current_epoch < self.start_epoch):
            return
        elif (current_epoch + 1) % self.epoch_interval == 0:
            checkpoint_path = training_process.save_checkpoint(
                    prefix=self.checkpoint_prefix
                    )
            
            if self.verbose:
                ts = datetime.now().strftime(self._ts_fmt)
                print(f"[{ts}] Scheduled checkpoint saved to "
                      f"{checkpoint_path}")

    def on_overfit_val_epoch_end(self, training_process):
        current_epoch = training_process.current_epoch

        if (current_epoch + 1) == training_process.max_epochs:
            checkpoint_path = training_process.save_checkpoint(
                    prefix=self.checkpoint_prefix
                    )

        if self.verbose:
            ts = datetime.now().strftime(self._ts_fmt)
            print(f"[{ts}] Last epoch checkpoint saved to "
                  f"{checkpoint_path}")


class BestCheckpointCallback(TrainingProcessCallback):
    """ Callback to save the best checkpoints based on a given metric direction.

    n_best (int): Number of checkpoints that are preserved.
    checkpoint_prefix (str): Prefix prepended to each checkpoint.
    metric (str): Name of the metric to be tracked. It must exactly match one
        of the metrics reported by a TrainingProcess instance.
    direction (str): Change in metric to be tracked (e.g. If min, the minimum
        value will be considered the best).
    verbose (str): If True, verbose mode is enabled.
    """
    def __init__(self,
                 n_best: int = 3,
                 checkpoint_prefix: str = 'best_',
                 metric: str = "avg_val_loss",
                 direction: str = "min",
                 verbose: bool = False):
        super().__init__()
        self.n_best = n_best
        self._session_history = []  # keeps (metric, filepath) tuples
        self.checkpoint_prefix = checkpoint_prefix
        self.metric = metric
        self.direction = direction
        self._ts_fmt = "%Y-%m-%d %H:%M:%S"
        self.verbose = verbose

    def _evaluate_metric(self, metric):
        if len(self._session_history) < self.n_best:
            return True

        session_metrics = [
                checkpoint_data[0] for checkpoint_data in self._session_history
                ]

        if self.direction == "min" and all(
                session_metric > metric for session_metric in session_metrics
                ):
            return True

        elif self.direction == "max" and all(
                session_metric > metric for session_metric in session_metrics
                ):
            raise NotImplementedError
        else:
            return False

    def on_val_epoch_end(self, training_process):
        # get metric
        metric = training_process.running_dict.get_last_value("avg_val_loss")
        epoch = training_process.current_epoch

        if self._evaluate_metric(metric):
            checkpoint_path = training_process.save_checkpoint(
                    prefix=self.checkpoint_prefix
                    )

            # log to console
            if self.verbose:
                ts = datetime.now().strftime(self._ts_fmt)
                print(f"[{ts}] Best checkpoint saved to "
                      f"{checkpoint_path}")

            # most recent element always goes at the beginning
            self._session_history.insert(0, (metric, checkpoint_path))

            if len(self._session_history) > self.n_best:
                checkpoint_file_to_delete = self._session_history[-1][1]
                os.remove(checkpoint_file_to_delete)
                del self._session_history[-1]

                if self.verbose:
                    ts = datetime.now().strftime(self._ts_fmt)
                    print(f"[{ts}] Old best checkpoint deleted from "
                          f"{checkpoint_file_to_delete}")


class AudioProcessTrackerCallback(TrainingProcessCallback):
    """ This callback allows to render spectrograms or predicted audios
    to tensorboard at intermediate steps during the network training process.

    IMPORTANT: It is assumed that both source and target files have the exact
    same name, although they can be (and should be) in different directories.

    input_dir (str): Input directory containing source files.
    output_dir (str): Output directory containing target files.
    epoch_interval (int): Number of epochs between saved intervals. 
    overfit_epoch_interval (int): Epoch interval used in overfit mode.
    sample_original_files (bool): If True, original files are also rendered
        to make it easier to compare them against predicted files.
    sample_rate (int): Sample rate of files to be predicted.
    sample_duration (int): Duration in seconds of each file.
    log_audio_file (bool): If True, playable audio files are added to 
        tensorboard.
    log_waveform_fig (bool): If True, a waveform plot is added to tensorboard.
    log_linspec_fig (bool): If True, a linear spectrogram is added to 
        tensorboard.
    log_logspec_fig (bool): If True, a log spectrogram is added to tensorboard.
    window_size (int): Window size (and FFT size) used for spectral analysis.
    hop_size (int): Hop size used for spectral analysis.
    pred_fn Union[Callable, str]: Callable of str of a function name to be used
        to predict the audio output.
    """
    def __init__(self,
                 input_dir: str = None,
                 output_dir: str = None,
                 epoch_interval: int = 25,
                 overfit_epoch_interval: int = 500, # to be implemented
                 sample_original_files: bool = True,
                 sample_rate: int = 16000,
                 sample_duration: int = 10,
                 log_audio_file: bool = True,
                 log_waveform_fig: bool = True,
                 log_linspec_fig: bool = True,
                 log_logspec_fig: bool = True,
                 window_size: int = 320,
                 hop_size: int = 160,
                 pred_fn: Callable = None):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.epoch_interval = epoch_interval
        self.sample_original_files = sample_original_files
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.log_audio_file = log_audio_file
        self.log_waveform_fig = log_waveform_fig
        self.log_linspec_fig = log_linspec_fig
        self.log_logspec_fig = log_logspec_fig
        self.window_size = window_size
        self.hop_size = hop_size
        self.pred_fn = pred_fn
        self.input_files, self.output_files = self._collect_files()

    def _collect_files(self, ext="*.wav"):
        """ Collects all files with extentions ext inside a folder. 

        Args:
            ext (str): Expression to be used within glob to collect all files
                of a given audio format.

        Returns:
            sorted_input_files, sorted_output_files (tuple): Tuple containing
                all input files and all output files alphabetically sorted.
        """
        input_files = glob.glob(os.path.join(self.input_dir, ext))
        output_files = glob.glob(os.path.join(self.output_dir, ext))
        return sorted(input_files), sorted(output_files)

    def _log_audio_file(self, logger, tag, audio_data, epoch, sample_rate):
        """ Logs an audio file to tensorboard.
        Args:
            logger (SummaryWritter): Logger object.
            tag (str): Tag to identify the rendered audio file.
            audio_data (np.array): Array containing the audio data.
            epoch (int): Current epoch number.
            sample_rate (int): Audio sample rate.
        """
        logger.add_audio(tag, audio_data, epoch, sample_rate)

    def _log_waveform_fig(self, logger, tag, audio_data, epoch, sample_rate):
        """ Logs a waveform plot figure to tensorboard. 
        Args:
            logger (SummaryWriter): logger object.
            tag (str): Tag to identify the rendered plot.
            audio_data (np.array): Array containing the audio data.
            epoch (int): Current epoch number.
            sample_rate (int): Audio sample rate.
        """

        # time axis
        t_ax = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))

        fig, ax = plt.subplots()
        ax.set_title(tag)
        ax.set_xlim([0, len(audio_data) / sample_rate])
        ax.set_ylim([-1.0, 1.0])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.plot(t_ax, audio_data)

        logger.add_figure(tag, fig, epoch)

    def _log_spectrogram_fig(self, logger, tag, audio_data,
                             epoch, sample_rate, 
                             window_size=2048, hop_size=512,
                             x_axis="time", y_axis="linear", fmt="%+.2d dB"):
        """Logs a spectrogram plot figure to tensorboard.
        Args:
            logger (SummaryWriter): Logger object. 
            tag (str): Tag to identify the rendered plot.
            audio_data (np.array): Array containing the audio data.
            epoch (int): Current epoch number.
            sample_rate (int): Audio sample rate.
            window_size (int): Window size (and FFT size) used for analysis.
            hop_size (int): Hop size used for analysis.
            x_axis (str): X axis librosa type.
            y_axis (str): Y axis librosa type.
            fmt (str): Format for y axis display.
        """
        audio_data_stft = librosa.stft(audio_data,
                                       n_fft=window_size,
                                       hop_length=hop_size)
        audio_data_stft_db = librosa.amplitude_to_db(np.abs(audio_data_stft),
                                                     ref=np.max)

        fig, ax = plt.subplots()
        img = librosa.display.specshow(audio_data_stft_db,
                                       x_axis=x_axis,
                                       y_axis=y_axis,
                                       ax=ax,
                                       sr=sample_rate,
                                       hop_length=hop_size)
        ax.set_title(tag)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(img, ax=ax, format=fmt)
        logger.add_figure(tag, fig, epoch)

    def on_val_epoch_end(self, training_process):
        current_epoch = training_process.current_epoch
        logger = training_process.logger

        if current_epoch == 0:
            for idx, input_file in enumerate(self.input_files):
                audio_data, _ = soundfile.read(
                        input_file,
                        frames=(self.sample_rate * self.sample_duration),
                        dtype="float32"
                        )

                if self.log_audio_file:
                    self._log_audio_file(logger, 
                                    f"Initial/source_{idx}",
                                    audio_data,
                                    current_epoch,
                                    self.sample_rate)

                if self.log_waveform_fig:
                    self._log_waveform_fig(logger,
                                           f"Initial/source_{idx}.waveform",
                                           audio_data,
                                           current_epoch,
                                           self.sample_rate)

                if self.log_linspec_fig:
                    self._log_spectrogram_fig(logger,
                                             f"Initial/source_{idx}.linspec",
                                             audio_data,
                                             current_epoch,
                                             self.sample_rate)

                if self.log_logspec_fig:
                    self._log_spectrogram_fig(logger,
                                             f"Initial/source_{idx}.logspec",
                                             audio_data,
                                             current_epoch,
                                             self.sample_rate,
                                             y_axis="log")

            for idx, output_file in enumerate(self.output_files):
                audio_data, _ = soundfile.read(
                        output_file,
                        frames=(self.sample_rate * self.sample_duration),
                        dtype="float32"
                        )

                if self.log_audio_file:
                    self._log_audio_file(logger,
                                    f"Initial/target_{idx}",
                                    audio_data,
                                    current_epoch,
                                    self.sample_rate)

                if self.log_waveform_fig:
                    self._log_waveform_fig(logger,
                                           f"Initial/target_{idx}.waveform",
                                           audio_data,
                                           current_epoch,
                                           self.sample_rate)

                if self.log_linspec_fig:
                    self._log_spectrogram_fig(logger,
                                              f"Initial/target_{idx}.linspec",
                                              audio_data,
                                              current_epoch,
                                              self.sample_rate)

                if self.log_logspec_fig:
                    self._log_spectrogram_fig(logger,
                                              f"Initial/target_{idx}.logspec",
                                              audio_data,
                                              current_epoch,
                                              self.sample_rate,
                                              y_axis="log")

        if (current_epoch == 0 or
            (current_epoch + 1) % self.epoch_interval == 0):
            
            for idx, input_file in enumerate(self.input_files):
                audio_data, _ = soundfile.read(
                        input_file,
                        frames=(self.sample_rate * self.sample_duration),
                        dtype="float32"
                        )
                
                # add reshaping to external process
                audio_data = audio_data.reshape(1, -1)
                audio_data = torch.from_numpy(audio_data) \
                                  .to(training_process.device)

                if callable(self.pred_fn):
                    pred_audio_data = self.pred_fn(audio_data)
                elif self.pred_fn is not None:
                    pred_audio_data = self.pred_fn(audio_data, training_process)
                else:
                    raise NotImplementedError(
                            "A prediction function is required"
                            )

                if self.log_audio_file:
                    self._log_audio_file(logger,
                                         f"Predicted/result_{idx}",
                                         pred_audio_data,
                                         current_epoch,
                                         self.sample_rate)

                if self.log_waveform_fig:
                    self._log_waveform_fig(logger,
                                           f"Predicted/result_{idx}.waveform",
                                           pred_audio_data,
                                           current_epoch,
                                           self.sample_rate)

                if self.log_linspec_fig:
                    self._log_spectrogram_fig(logger,
                                             f"Predicted/result_{idx}.linspec",
                                             pred_audio_data,
                                             current_epoch,
                                             self.sample_rate)

                if self.log_logspec_fig:
                    self._log_spectrogram_fig(logger,
                                              f"Predicted/result_{idx}.logspec",
                                              pred_audio_data,
                                              current_epoch,
                                              self.sample_rate,
                                              y_axis="log")

    def on_overfit_val_epoch_end(self, training_process):
        self.on_val_epoch_end(training_process)
