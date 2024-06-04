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

# Create models
english_model = mutlimodal(args).to(rank)

if rank == 0: heading(f'\nSetting up image model ')
image_model = vision(args).to(rank)

if rank == 0: heading(f'\nSetting up attention model ')
attention = ScoringAttentionModule(args).to(rank)

if rank == 0: heading(f'\nSetting up contrastive loss ')
contrastive_loss = ContrastiveLoss(args).to(rank)


model_with_params_to_update = {
    "enlish_model": english_model,
    "attention": attention,
    "contrastive_loss": contrastive_loss,
    "image_model": image_model
    }
model_to_freeze = {
    }
trainable_parameters = getParameters(model_with_params_to_update, model_to_freeze, args)

if args["optimizer"] == 'sgd':
    optimizer = torch.optim.SGD(
        trainable_parameters, args["learning_rate_scheduler"]["initial_learning_rate"],
        momentum=args["momentum"], weight_decay=args["weight_decay"]
        )
elif args["optimizer"] == 'adam':
    optimizer = torch.optim.Adam(
        trainable_parameters, args["learning_rate_scheduler"]["initial_learning_rate"],
        weight_decay=args["weight_decay"]
        )
else:
    raise ValueError('Optimizer %s is not supported' % args["optimizer"])


english_model = DDP(english_model, device_ids=[rank])
image_model = DDP(image_model, device_ids=[rank]) 

if "restore_epoch" in args:
    info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAtEpochAMP(
        args["exp_dir"], english_model, image_model, attention, contrastive_loss, optimizer, rank, 
        args["restore_epoch"]
        )
else: 
    heading(f'\nRetoring model parameters from best epoch ')
    info, epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAMP(
        args["exp_dir"], english_model, image_model, attention, contrastive_loss, optimizer, 
        rank, False
        )

episodes = np.load(Path('data/episodes.npz'), allow_pickle=True)['episodes'].item()

one_familiar_results = {'correct': 0, 'total': 0}
two_familiar_results = {'correct': 0, 'total': 0}
novel_results = {'correct': 0, 'total': 0}
known_novel_results = {'correct': 0, 'total': 0}
per_novel_word = {}
per_familiar_word = {}
per_novel_word_faults = {}
print(f'\n\n')
rewind = "\033[A"*2
c = 0
t = 0
with torch.no_grad():

    audio_datapoints = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_datapoints'].item()
    image_datapoints = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_datapoints'].item()

    query = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_1'].item()
    query_labels = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_labels_1'].item()
    query_tag = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_tag_1'].item()

    secondary_query = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_2'].item()
    secondary_labels = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_labels_2'].item()
    secondary_tag = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_tag_2'].item()

    image_1 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_1'].item()
    image_2 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_2'].item()
    image_labels_1 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_labels_1'].item()
    image_labels_2 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_labels_2'].item()
    image_tag_1 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_tag_1'].item()
    image_tag_2 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_tag_2'].item()

    # episodes = np.load(Path('data/episodes.npz'), allow_pickle=True)['episodes'].item()

    for ep_num in tqdm(episodes, desc=f'{rewind}'):
        # if ep_num != 25: continue
        episode = episodes[ep_num]

        tags_1 = query_tag[ep_num]
        labels_1 = query_labels[ep_num]
        tags_2 = secondary_tag[ep_num]
        labels_2 = secondary_labels[ep_num]

   
        for i in range(len(tags_1)):

            index = np.random.randint(0, 2)

            if tags_1[i] == 'familiar_1':
                if labels_1[i] not in per_familiar_word: per_familiar_word[labels_1[i]] = {'correct': 0, 'total': 0}
                if index == 0: 
                    one_familiar_results['correct'] += 1
                    per_familiar_word[labels_1[i]]['correct'] += 1
                one_familiar_results['total'] += 1
                per_familiar_word[labels_1[i]]['total'] += 1

            elif tags_1[i] == 'novel':
                if labels_1[i] not in per_novel_word: per_novel_word[labels_1[i]] = {'correct': 0, 'total': 0}
                if labels_1[i] not in per_novel_word_faults: per_novel_word_faults[labels_1[i]] = {}
                if index == 0: 
                    novel_results['correct'] += 1
                    per_novel_word[labels_1[i]]['correct'] += 1
                    c += 1
                else:
                    if labels_2[i] not in per_novel_word_faults[labels_1[i]]: per_novel_word_faults[labels_1[i]][labels_2[i]] = 0
                    per_novel_word_faults[labels_1[i]][labels_2[i]] += 1
                novel_results['total'] += 1
                per_novel_word[labels_1[i]]['total'] += 1
                t += 1

            index = np.random.randint(0, 2)

            if tags_2[i] == 'familiar_2':
                if labels_2[i] not in per_familiar_word: per_familiar_word[labels_2[i]] = {'correct': 0, 'total': 0}
                if index == 1: 
                    two_familiar_results['correct'] += 1
                    per_familiar_word[labels_2[i]]['correct'] += 1
                two_familiar_results['total'] += 1
                per_familiar_word[labels_2[i]]['total'] += 1
            elif tags_2[i] == 'known_novel':
                if index == 1: known_novel_results['correct'] += 1
                known_novel_results['total'] += 1
        print(f'{c}/{t}={100*c/t}%')

        # if ep_num == 9: break
        # break


results = {
    'familiar_1': one_familiar_results, 
    'familiar_2': two_familiar_results, 
    'novel': novel_results, 
    'known_novel': known_novel_results}
for r in results:
    d = results[r]
    c = d['correct']
    t = d['total']
    p = 100*c/t
    print(f'{r:<12}: {c}/{t}={p:.2f}%')

name = 'keyword'
if args['cpc']['warm_start']: name += '_cpc'
if args['pretrained_alexnet']: name += '_alexnet'

f = open(Path(f'results/{name}.txt'), 'w')

for w in per_novel_word:

    d = per_novel_word[w]
    c = d['correct']
    t = d['total']
    p = 100*c/t
    print(f'{w:<12}: {c}/{t}={p:.2f}%')
    f.write(f'{w:<12} {p}\n')

# print(f'Novel faults')
# for w in per_novel_word_faults:
#     d = per_novel_word_faults[w]
#     t = per_novel_word[w]['total']
#     for con in d:
#         p = 100*d[con]/t
#         print(f'{w:<12} -> {con:<12}: {d[con]}/{t}={p:.2f}')

# print(f'\nFamiliar accuarcies')
# for w in per_familiar_word:
#     d = per_familiar_word[w]
#     c = d['correct']
#     t = d['total']
#     p = 100*c/t
#     print(f'{w:<12}: {c}/{t}={p:.2f}%')