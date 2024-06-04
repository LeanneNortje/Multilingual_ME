#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import json
import librosa
import numpy as np
import os
from PIL import Image
import scipy.signal
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchaudio
import scipy
from pathlib import Path

scipy_windows = {
    'hamming': scipy.signal.hamming,
    'hann': scipy.signal.hann, 
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett
    }

def preemphasis(x,coeff=0.97):  
    # function adapted from https://github.com/dharwath
    return scipy.signal.lfilter([1, -coeff], [1], x)
    # return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class spokenData(Dataset):

    def __init__(self, data, audio_fn, audio_conf, add_support_set=False):

        self.data = data
        self.audio_fn = Path(audio_fn)
        self.audio_conf = audio_conf

        self.wav_dict = {}

        fns = self.audio_fn.rglob("*.wav")
        for fn in fns:
            if fn.stem not in self.wav_dict: self.wav_dict[fn.stem] = fn


    def _LoadAudio(self, path):

        audio_type = self.audio_conf.get('audio_type')
        if audio_type not in ['melspectrogram', 'spectrogram']:
            raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')
        
        preemph_coef = self.audio_conf.get('preemph_coef')
        sample_rate = self.audio_conf.get('sample_rate')
        window_size = self.audio_conf.get('window_size')
        window_stride = self.audio_conf.get('window_stride')
        window_type = self.audio_conf.get('window_type')
        num_mel_bins = self.audio_conf.get('num_mel_bins')
        target_length = self.audio_conf.get('target_length')
        fmin = self.audio_conf.get('fmin')
        n_fft = self.audio_conf.get('n_fft', int(sample_rate * window_size))
        win_length = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)

        # load audio, subtract DC, preemphasis
        _, sample_rate = torchaudio.load(path)
        y, sr = librosa.load(path, sample_rate)
        if y.size == 0:
            y = np.zeros(target_length)
        y = y - y.mean()
        y = preemphasis(y, preemph_coef)

        # compute mel spectrogram / filterbanks
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=scipy_windows.get(window_type, scipy_windows['hamming']))
        spec = np.abs(stft)**2 # Power spectrum
        if audio_type == 'melspectrogram':
            mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
            melspec = np.dot(mel_basis, spec)
            logspec = librosa.power_to_db(melspec, ref=np.max)
        elif audio_type == 'spectrogram':
            logspec = librosa.power_to_db(spec, ref=np.max)
        n_frames = logspec.shape[1]
        logspec = torch.FloatTensor(logspec)
        return logspec, n_frames
        
    def __getitem__(self, index):
        datum = self.data[index]
        # print(Path(datum['wav']))
        wav_fn = self.wav_dict[Path(datum['wav']).stem]
        ids = datum['ids']
        audio_feat, frames = self._LoadAudio(wav_fn)
        wav_name = Path(datum['wav']).stem
        ids = [str(id) for id in ids]

        return audio_feat, wav_name, ids, frames

    def __len__(self):
        return len(self.data)