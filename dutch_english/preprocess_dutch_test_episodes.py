#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2023
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
# adapted from https://github.com/dharwath

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel as DDP
from dataloaders import *
from models.setup import *
from models.util import *
from models.GeneralModels import *
from models.multimodalModels import *
from training.util import *
from evaluation.calculations import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from training import validate
import time
from tqdm import tqdm

import numpy as trainable_parameters
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy
import scipy.signal
from scipy.spatial import distance
import librosa
import matplotlib.lines as lines

import itertools
import seaborn as sns
from torchvision.io import read_image
from torchvision.models import *

from PIL import Image
from matplotlib import image
from matplotlib import pyplot

BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"

flickr_boundaries_fn = Path('/storage/Datasets/flickr_audio/flickr_8k.ctm')
flickr_audio_dir = flickr_boundaries_fn.parent / "wavs"
flickr_images_fn = Path('/storage/Datasets/Flicker8k_Dataset/')
flickr_segs_fn = Path('./data/flickr_image_masks/')

config_library = {
    "multilingual": "English_Hindi_DAVEnet_config.json",
    "multilingual+matchmap": "English_Hindi_matchmap_DAVEnet_config.json",
    "english": "English_DAVEnet_config.json",
    "english+matchmap": "English_matchmap_DAVEnet_config.json",
    "hindi": "Hindi_DAVEnet_config.json",
    "hindi+matchmap": "Hindi_matchmap_DAVEnet_config.json",
}

scipy_windows = {
    'hamming': scipy.signal.hamming,
    'hann': scipy.signal.hann, 
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett
    }

def myRandomCrop(im, resize, to_tensor):

        im = resize(im)
        im = to_tensor(im)
        return im

def preemphasis(signal,coeff=0.97):  
    # function adapted from https://github.com/dharwath
    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

def LoadAudio(path, audio_conf):
    
    audio_type = audio_conf.get('audio_type')
    if audio_type not in ['melspectrogram', 'spectrogram']:
        raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')

    preemph_coef = audio_conf.get('preemph_coef')
    sample_rate = audio_conf.get('sample_rate')
    window_size = audio_conf.get('window_size')
    window_stride = audio_conf.get('window_stride')
    window_type = audio_conf.get('window_type')
    num_mel_bins = audio_conf.get('num_mel_bins')
    target_length = audio_conf.get('target_length')
    fmin = audio_conf.get('fmin')
    n_fft = audio_conf.get('n_fft', int(sample_rate * window_size))
    win_length = int(sample_rate * window_size)
    hop_length = int(sample_rate * window_stride)

    # load audio, subtract DC, preemphasis
    y, sr = librosa.load(path, sample_rate)
    dur = librosa.get_duration(y=y, sr=sr)
    nsamples = y.shape[0]
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
    # n_frames = logspec.shape[1]
    logspec = torch.FloatTensor(logspec)
    nsamples = logspec.size(1)

    # y, sr = librosa.load(path, mono=True)
    # print(y, '\n')
    # y = torch.tensor(y).unsqueeze(0)
    # nsamples = y.size(1)
    # print(y.size(), '\n')

    return logspec, nsamples#, n_frames

def LoadImage(impath, resize, image_normalize, to_tensor):
    img = Image.open(impath).convert('RGB')
    # img = self.image_resize_and_crop(img)
    img = myRandomCrop(img, resize, to_tensor)
    img = image_normalize(img)
    return img

def PadFeat(feat, target_length, padval):
    # feat = feat.transpose(1, 2)
    # print(feat.size())
    nframes = feat.size(1)
    pad = target_length - nframes

    if pad > 0:
        feat = np.pad(feat, ((0, 0), (0, pad)), 'constant',
            constant_values=(padval, padval))
    elif pad < 0:
        nframes = target_length
        feat = feat[:, 0: pad]

    feat = torch.tensor(feat).unsqueeze(0)
    # print(feat.size())
    return feat, torch.tensor(nframes).unsqueeze(0)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--resume", action="store_true", dest="resume",
        help="load from exp_dir if True")
parser.add_argument("--config-file", type=str, default='matchmap', choices=['matchmap'], help="Model config file.")
parser.add_argument("--restore-epoch", type=int, default=-1, help="Epoch to generate accuracies for.")
parser.add_argument("--image-base", default="/storage", help="Model config file.")
command_line_args = parser.parse_args()
restore_epoch = command_line_args.restore_epoch

# Setting up model specifics
heading(f'\nSetting up model files ')
args, image_base = modelSetup(command_line_args, True)
rank = 'cuda'

audio_conf = args["audio_config"]
target_length = audio_conf.get('target_length', 128)
padval = audio_conf.get('padval', 0)
image_conf = args["image_config"]
crop_size = image_conf.get('crop_size')
center_crop = image_conf.get('center_crop')
RGB_mean = image_conf.get('RGB_mean')
RGB_std = image_conf.get('RGB_std')

# image_resize_and_crop = transforms.Compose(
#         [transforms.Resize(224), transforms.ToTensor()])
resize = transforms.Resize((256, 256))
to_tensor = transforms.ToTensor()
image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

image_resize = transforms.transforms.Resize((256, 256))
trans = transforms.ToPILImage()


episodes = np.load(Path('data/multilingual_episodes.npz'), allow_pickle=True)['episodes'].item()

dutch_audio_1 = {}
dutch_audio_labels_1 = {}
dutch_audio_tag_1 = {}
dutch_audio_2 = {}
dutch_audio_labels_2 = {}
dutch_audio_tag_2 = {}
image_1 = {}
image_labels_1 = {}
image_tag_1 = {}
image_dataset_1 = {}
image_2 = {}
image_labels_2 = {}
image_tag_2 = {}
image_dataset_2 = {}
audio_datapoints = {}
image_datapoints = {}

with torch.no_grad():

    for ep_num in tqdm(episodes):

        episode = episodes[ep_num]

        # Familiar
#         q = []
#         q_nframes = []
#         one = []

#         # Novel
        image_1[ep_num] = []
        image_2[ep_num] = []
        image_labels_1[ep_num] = []
        image_labels_2[ep_num] = []
        image_tag_1[ep_num] = []
        image_tag_2[ep_num] = []
        image_dataset_1[ep_num] = []
        image_dataset_2[ep_num] = []

        dutch_audio_1[ep_num] = []
        dutch_audio_2[ep_num] = []
        dutch_audio_labels_1[ep_num] = []
        dutch_audio_labels_2[ep_num] = []
        dutch_audio_tag_1[ep_num] = []
        dutch_audio_tag_2[ep_num] = []

        # french_audio_1[ep_num] = []
        # french_audio_2[ep_num] = []
        # french_audio_labels_1[ep_num] = []
        # french_audio_labels_2[ep_num] = []
        # french_audio_tag_1[ep_num] = []
        # french_audio_tag_2[ep_num] = []

        for test_class, imgpath, wav, dutch_wav, french_wav, dataset in episode['novel_test']['novel']:
            if imgpath not in image_datapoints:
                this_image = LoadImage(imgpath, resize, image_normalize, to_tensor)
                image_datapoints[imgpath] = this_image.unsqueeze(0)
            image_labels_1[ep_num].append(test_class)
            image_1[ep_num].append(imgpath)
            image_tag_1[ep_num].append('novel')
            image_dataset_1[ep_num].append(dataset)

            if dutch_wav not in audio_datapoints:
                this_audio_feat, this_nframes = LoadAudio(dutch_wav, audio_conf)
                this_audio_feat, this_nframes = PadFeat(this_audio_feat, target_length, padval)
                audio_datapoints[dutch_wav] = this_audio_feat
            dutch_audio_labels_1[ep_num].append(test_class)
            dutch_audio_1[ep_num].append(dutch_wav)
            dutch_audio_tag_1[ep_num].append('novel')

            # if french_wav not in audio_datapoints:
            #     this_audio_feat, this_nframes = LoadAudio(french_wav, audio_conf)
            #     this_audio_feat, this_nframes = PadFeat(this_audio_feat, target_length, padval)
            #     audio_datapoints[french_wav] = this_audio_feat
            # french_audio_labels_1[ep_num].append(test_class)
            # french_audio_1[ep_num].append(french_wav)
            # french_audio_tag_1[ep_num].append('novel')

        for known_class, imgpath, wav, dutch_wav, french_wav, dataset in episode['novel_test']['known']:
            
            if imgpath not in image_datapoints:
                this_image = LoadImage(imgpath, resize, image_normalize, to_tensor)
                image_datapoints[imgpath] = this_image.unsqueeze(0)
            image_labels_2[ep_num].append(known_class)
            image_2[ep_num].append(imgpath)
            image_tag_2[ep_num].append('known_novel')
            image_dataset_2[ep_num].append(dataset)

            if dutch_wav not in audio_datapoints:
                this_audio_feat, this_nframes = LoadAudio(dutch_wav, audio_conf)
                this_audio_feat, this_nframes = PadFeat(this_audio_feat, target_length, padval)
                audio_datapoints[dutch_wav] = this_audio_feat
            dutch_audio_labels_2[ep_num].append(known_class)
            dutch_audio_2[ep_num].append(dutch_wav)  
            dutch_audio_tag_2[ep_num].append('known_novel')

            # if french_wav not in audio_datapoints:
            #     this_audio_feat, this_nframes = LoadAudio(french_wav, audio_conf)
            #     this_audio_feat, this_nframes = PadFeat(this_audio_feat, target_length, padval)
            #     audio_datapoints[french_wav] = this_audio_feat
            # french_audio_labels_2[ep_num].append(known_class)
            # french_audio_2[ep_num].append(french_wav)  
            # french_audio_tag_2[ep_num].append('known_novel')


        for test_class, imgpath, wav, dutch_wav, french_wav, dataset in episode['familiar_test']['test']:
            if imgpath not in image_datapoints:
                this_image = LoadImage(imgpath, resize, image_normalize, to_tensor)
                image_datapoints[imgpath] = this_image.unsqueeze(0)
            image_labels_1[ep_num].append(test_class)
            image_1[ep_num].append(imgpath)
            image_tag_1[ep_num].append('familiar_1')
            image_dataset_1[ep_num].append(dataset)

            if dutch_wav not in audio_datapoints:
                this_audio_feat, this_nframes = LoadAudio(dutch_wav, audio_conf)
                this_audio_feat, this_nframes = PadFeat(this_audio_feat, target_length, padval)
                audio_datapoints[dutch_wav] = this_audio_feat
            dutch_audio_labels_1[ep_num].append(test_class)
            dutch_audio_1[ep_num].append(dutch_wav)
            dutch_audio_tag_1[ep_num].append('familiar_1')

            # if french_wav not in audio_datapoints:
            #     this_audio_feat, this_nframes = LoadAudio(french_wav, audio_conf)
            #     this_audio_feat, this_nframes = PadFeat(this_audio_feat, target_length, padval)
            #     audio_datapoints[french_wav] = this_audio_feat
            # french_audio_labels_1[ep_num].append(test_class)
            # french_audio_1[ep_num].append(french_wav)
            # french_audio_tag_1[ep_num].append('familiar_1')


        for known_class, imgpath, wav, dutch_wav, french_wav, dataset in episode['familiar_test']['other']:
            
            if imgpath not in image_datapoints:
                this_image = LoadImage(imgpath, resize, image_normalize, to_tensor)
                image_datapoints[imgpath] = this_image.unsqueeze(0)
            image_labels_2[ep_num].append(known_class)
            image_2[ep_num].append(imgpath)
            image_tag_2[ep_num].append('familiar_2')
            image_dataset_2[ep_num].append(dataset)

            if dutch_wav not in audio_datapoints:
                this_audio_feat, this_nframes = LoadAudio(dutch_wav, audio_conf)
                this_audio_feat, this_nframes = PadFeat(this_audio_feat, target_length, padval)
                audio_datapoints[dutch_wav] = this_audio_feat
            dutch_audio_labels_2[ep_num].append(known_class)
            dutch_audio_2[ep_num].append(dutch_wav)  
            dutch_audio_tag_2[ep_num].append('familiar_2')

            # if french_wav not in audio_datapoints:
            #     this_audio_feat, this_nframes = LoadAudio(french_wav, audio_conf)
            #     this_audio_feat, this_nframes = PadFeat(this_audio_feat, target_length, padval)
            #     audio_datapoints[french_wav] = this_audio_feat
            # french_audio_labels_2[ep_num].append(known_class)
            # french_audio_2[ep_num].append(french_wav)  
            # french_audio_tag_2[ep_num].append('familiar_2') 

save_fn = Path('results/files/dutch_episode_data')
save_fn.parent.mkdir(exist_ok=True, parents=True)
np.savez(
    save_fn,
    audio_datapoints=audio_datapoints,
    image_datapoints=image_datapoints,
    audio_1=dutch_audio_1,
    audio_labels_1=dutch_audio_labels_1,
    audio_tag_1=dutch_audio_tag_1,
    audio_2=dutch_audio_2,
    audio_labels_2=dutch_audio_labels_2,
    audio_tag_2=dutch_audio_tag_2,
    image_1=image_1,
    image_labels_1=image_labels_1,
    image_tag_1=image_tag_1,
    image_dataset_1=image_dataset_1,
    image_2=image_2,
    image_labels_2=image_labels_2,
    image_tag_2=image_tag_2, 
    image_dataset_2=image_dataset_2
    )