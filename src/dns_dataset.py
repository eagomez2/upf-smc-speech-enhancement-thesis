import os
import glob
import wavinfo
import soundfile
import librosa
import tqdm
import torch
import torch.utils.data
import numpy as np


class DNSDataset(torch.utils.data.Dataset):
    """ Microsoft Deep Noise Suppression dataset. 

    This dataset contains a series of clean and noisy speech pairs provided
    as part of the Deep Noise Suppression Challenge by Microsoft:

    https://github.com/microsoft/DNS-Challenge/tree/master/datasets

    All samples are assumed to be not shorter than 30 seconds. Each noisy
    speech file is stored in output_dir and the corresponding clean speech
    file is stored in input_dir. The input/output pairs may differ in name
    but they have their respective fileid in common, which is also the criteria 
    used for sorting to avoid mismatches between files.

    Args:
        input_dir (str): Input directory containing noisy speech files.
        output_dir (str): Output directory containing clean speech files.
        sample_rate (int): Sample rate of audio files in input_dir and
            output_dir.
        chl_n (int): Number of channels of each audio file.
        bit_depth (int): Bit depth of each audio file.
        sample_duration (int): Sample duration in seconds of each chunk to
        output while iterating through this dataset.
        window_size (int): Window size (and FFT size) used for analysis.
        hop_size (int): Hop size used for analysis.
        stft (bool): If True, the STFT of the audio is output instead of the
            raw audio data.
    """
    def __init__(self,
                 input_dir: str = "dataset/example_dataset/noisy_speech",
                 output_dir: str = "dataset/example_dataset/clean_speech",
                 sample_rate: int = 16000,
                 chl_n: int = 1,
                 bit_depth: int = 16,
                 sample_duration: int = 10,
                 window_size: int = 320,
                 hop_size: int = 160,
                 stft: bool = False):
        
        super().__init__()

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.chl_n = chl_n
        self.bit_depth = bit_depth
        self.sample_duration = sample_duration
        self._chunk_samples = self.sample_rate * self.sample_duration
        self.input_files, self.output_files = self._perform_integrity_check()
        self.stft = stft
        self.window_size = window_size
        self.hop_size = hop_size

    def get_fileid_n(self, filename: str):
        """ Helper function to sort files by fileid.

        Retrieves the fileid of each a file following the naming convention
        of the DNS Challenge where input files end with _fileid_N_, where N
        is a number of 1 or more digits. Matching output files present
        the same suffix. Final fileid is padded with zeros if it is shorter
        than 10 numbers to ensure the correct ordering of files even if there
        are many hours of audio.

        Args:
            filename (str): Name of the input or output file.

        Returns:
            file_id (int): Zero padded integer containing the fileid number
                of the input file.
        """
        filename = os.path.splitext(filename)[0]
        file_id = filename.split("_fileid_")[1].zfill(10)
        return file_id

    @property
    def info(self):
        return f"""
        Deep Noise Suppression Challenge Dataset.

        This dataset contains pairs of noisy speech files as input and
        matches clean speech files as outputs. Each file meets the following
        criteria:

        1. All files have {self.chl_n} channel(s)
        2. All files are {self.bit_depth} bits
        3. All files are sampled at {self.sample_rate}Hz
        4. All files are at least {self.sample_duration} seconds long
        """
    def _perform_integrity_check(self):
        """ Performs a series of steps to check if each file of the dataset
        meet a prescribed criteria. The following conditions are tested:

        IMPORTANT: Only .wav extension files are allowed.

        1. input_dir and output_dir exist and are not empty.
        2. input_dir and output_dir have the same number of input_files.
        3. All files have the same sample rate.
        4. All files have the same bit depth.
        5. All files have the same number of channels.
        6. All files are at least the same duration as sample_duration.
        7. Byte count of input files match byte count of output files.

        Returns:
            input_files, output_files (list, list)

        Raises:
            FileNotFoundError: If input_dir or output_dir do not exist or
                are empty.
            ValueError: If the number of input and output files, sample rates,
            bit depths, channel number, duration or bit count do not match.
        """
        if not os.path.isdir(self.input_dir):
            raise FileNotFoundError(f"input_dir not found: {self.input_dir}")

        if not os.path.isdir(self.output_dir):
            raise FileNotFoundError(f"output_dir not found: {self.output_dir}")

        input_files = sorted(glob.glob(os.path.join(self.input_dir, "*.wav")),
                             key=self.get_fileid_n)
        output_files = sorted(glob.glob(os.path.join(self.output_dir, "*.wav")),
                              key=self.get_fileid_n)

        if len(input_files) == 0:
            raise FileNotFoundError(f"input_dir is empty: {self.input_dir}")

        if len(output_files) == 0:
            raise FileNotFoundError(f"output_dir is empty: {self.output_dir}")

        if len(input_files) != len(output_files):
            raise ValueError(f"""
            input_dir and output_dir have a different number of files:
            input_dir: {len(input_files)} files found.
            output_dir: {len(output_files)} files found.
            """)

        for audio_file in tqdm.tqdm(
                (input_files + output_files),
                desc="Dataset integrity check 1/2", 
                leave=False,
                unit="files"
                ):
            audio_file_info = wavinfo.WavInfoReader(audio_file)
            sample_rate = audio_file_info.fmt.sample_rate
            chl_n = audio_file_info.fmt.channel_count
            bit_depth = audio_file_info.fmt.bits_per_sample
            duration = audio_file_info.data.frame_count / sample_rate

            if sample_rate != self.sample_rate:
                raise ValueError(f"""
                sample_rate not matching requirements ({self.sample_rate}):
                file: {audio_file}
                detected sample_rate: {sample_rate}
                """)

            if bit_depth != self.bit_depth:
                raise ValueError(f"""
                bit_depth not matching requirements ({self.bit_depth}):
                file: {audio_file}
                detected bit_depth: {bit_depth}
                """)

            if chl_n != self.chl_n:
                raise ValueError(f"""
                channels not matching requirements ({self.chl_n}):
                file: {audio_file}
                detected chls: {chls}
                """)

            if duration < self.sample_duration:
                raise ValueError(f"""
                duration not matching requirements ({self.sample_duration}s):
                file: {audio_file}
                detected duration: {duration}s
                """)

        for input_file, output_file in tqdm.tqdm(
                zip(input_files, output_files),
                desc="Dataset integrity check 2/2",
                leave=False,
                unit="files"
                ):
            input_file_byte_count = wavinfo.WavInfoReader(input_file) \
                                           .data.byte_count
            output_file_byte_count = wavinfo.WavInfoReader(output_file) \
                                            .data.byte_count

            if input_file_byte_count != output_file_byte_count:
                raise ValueError(f"""
                unmatched pair of files found:
                input_file: {input_file}
                input_file_byte_count: {input_file_byte_count}
                output_file: {output_file}
                ouput_file_byte_count: {output_file_byte_count}
                """)

        return input_files, output_files

    def __getitem__(self, idx: int):
        """ 
        Returns a sample of duration self.sample_duration of dtype=float32.
        The total numer of samples of each chunk is self._chunk_samples.

        Args:
            idx (int): Index of the input/output pair to be retrieved.

        Returns:
            noisy_speech_chunk, clean_speech_chunk (np.ndarray, np.ndarray):
                Tuple of np.ndarrays of float32 values corresponding to a
                noisy_speech_chunk and its matching clean_speech_chunk.
        """
        input_file, output_file = self.input_files[idx], self.output_files[idx]

        noisy_speech, _ = soundfile.read(input_file, dtype="float32")
        clean_speech, _ = soundfile.read(output_file, dtype="float32")

        noisy_speech_chunk = noisy_speech[0 : self._chunk_samples]
        clean_speech_chunk = clean_speech[0 : self._chunk_samples]

        if self.stft:
            # return x_complex, x_log_pow_mag, y_complex, y_log_pow_mag
            hann_window = torch.hann_window(self.window_size)

            noisy_speech = torch.from_numpy(noisy_speech_chunk)
            clean_speech = torch.from_numpy(clean_speech_chunk)

            noisy_complex = torch.stft(noisy_speech,
                                    onesided=True,
                                    n_fft=self.window_size,
                                    center=True,
                                    hop_length=self.hop_size,
                                    normalized=False,
                                    window=hann_window,
                                    return_complex=True)

            clean_complex = torch.stft(clean_speech,
                                       onesided=True,
                                       n_fft=self.window_size,
                                       center=True,
                                       hop_length=self.hop_size,
                                       normalized=False,
                                       window=hann_window,
                                       return_complex=True)

            noisy_log_pow_spec = torch.log10(torch.abs(noisy_complex)**2 + 1e-10)
            clean_log_pow_spec = torch.log10(torch.abs(clean_complex)**2 + 1e-10)

            return (
                    (noisy_complex, noisy_log_pow_spec), 
                    (clean_complex, clean_log_pow_spec)
                    )
        else:
            return noisy_speech_chunk, clean_speech_chunk

    def __len__(self):
        """ Returns the amount of input/output pairs in the dataset. 

        Returns:
            len of input/output pairs (int): Number of input/output pairs.
        """
        return len(self.input_files)
