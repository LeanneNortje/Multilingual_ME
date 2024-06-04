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
from models.infonce import infonce
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

episodes = np.load(Path('data/episodes.npz'), allow_pickle=True)['episodes'].item()
fam_1 = []
fam_2 = []
me = []
fam_me = []

print(f'\n\n')
rewind = "\033[A"*2
c = 0
t = 0
with torch.no_grad():

    audio_datapoints = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['audio_datapoints'].item()
    image_datapoints = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['image_datapoints'].item()

    query = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['audio_1'].item()
    query_labels = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['audio_labels_1'].item()
    query_tag = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['audio_tag_1'].item()

    secondary_query = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['audio_2'].item()
    secondary_labels = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['audio_labels_2'].item()
    secondary_tag = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['audio_tag_2'].item()

    image_1 = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['image_1'].item()
    image_2 = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['image_2'].item()
    image_labels_1 = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['image_labels_1'].item()
    image_labels_2 = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['image_labels_2'].item()
    image_tag_1 = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['image_tag_1'].item()
    image_tag_2 = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['image_tag_2'].item()

    # episodes = np.load(Path('data/episodes.npz'), allow_pickle=True)['episodes'].item()
    for args_fn in Path('model_metadata').rglob('*/args.pkl'):
        one_familiar_results = {'correct': 0, 'total': 0}
        two_familiar_results = {'correct': 0, 'total': 0}
        novel_results = {'correct': 0, 'total': 0}
        known_novel_results = {'correct': 0, 'total': 0}
        per_novel_word = {}
        per_familiar_word = {}
        per_novel_word_faults = {}

        with open(args_fn, "rb") as f:
            args = pickle.load(f)

            # Setting up model specifics
        # heading(f'\nSetting up model files ')
        # args, image_base = modelSetup(command_line_args, True)
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
        audio_model = mutlimodal(args).to(rank)
        image_model = vision(args).to(rank)
        attention = ScoringAttentionModule(args).to(rank)
        contrastive_loss = infonce
        model_with_params_to_update = {
            "audio_model": audio_model,
            "attention": attention,
            # "contrastive_loss": contrastive_loss,
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


        audio_model = DDP(audio_model, device_ids=[rank])
        image_model = DDP(image_model, device_ids=[rank]) 

        if "restore_epoch" in args:
            info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAtEpochAMP(
                args["exp_dir"], audio_model, image_model, attention, contrastive_loss, optimizer, rank, 
                args["restore_epoch"]
                )
        else: 
            heading(f'\nRetoring model parameters from best epoch ')
            info, epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAMP(
                args["exp_dir"], audio_model, image_model, attention, contrastive_loss, optimizer, 
                rank, False
                )
        novel_classes = set()

        for ep_num in tqdm(episodes):
            # if ep_num != 25: continue
            episode = episodes[ep_num]

            # print(len(episodes[ep_num]['novel_test']['novel']))
            query_output = []
            for i, name in enumerate(query[ep_num]):
                if query_tag[ep_num][i] == 'novel':
                    # print(query_tag[ep_num][i])
                    query_output.append(audio_datapoints[name])
                    novel_classes.add(query_labels[ep_num][i])
            query_output= torch.cat(query_output, dim=0)
            _, _, query_output = audio_model(query_output.to(rank))
            # break

            query_image = []
            for i, name in enumerate(image_1[ep_num]):
                if query_tag[ep_num][i] == 'novel':
                    query_image.append(image_datapoints[name])
            query_image = torch.cat(query_image, dim=0)
            query_image = image_model(query_image.to(rank))


            all_indices = [i for i in range(query_output.size(0))]
            
            a = all_indices[0: len(query_output) // 2]
            b = all_indices[len(query_output) // 2:]
            np.random.shuffle(b)
            query_a = query_output[a, :]
            query_b = query_output[b, :]
            image_a = query_image[a, :]
            image_b = query_image[b, :]
            query_labels_a = [query_labels[ep_num][a_i] for a_i in a]
            query_labels_b = [query_labels[ep_num][b_i] for b_i in b]

            for n_c in novel_classes:
                if n_c not in per_novel_word: per_novel_word[n_c] = {}
                for n_c2 in novel_classes:
                    if n_c2 not in per_novel_word[n_c]: per_novel_word[n_c][n_c2] = {'count': 0, 'encountered': 0}

            for n, label in enumerate(query_labels[ep_num]):
                if query_tag[ep_num][n] == 'novel' and label not in per_novel_word: per_novel_word[label] = {'correct': 0, 'total': 0}

            for i in range(query_a.size(0)):
                label_a = query_labels_a[i]
                label_b = query_labels_b[i]

                im = torch.cat([image_a[i, :, :].unsqueeze(0), image_b[i, :, :].unsqueeze(0)], dim=0)
                labels = [label_a, label_b]
                scores = attention.one_to_many_score(im, query_a[i, :, :].unsqueeze(0)).squeeze()
                index = torch.argmax(scores).item()

                if index == 0: 
                    novel_results['correct'] += 1
                    c += 1
                novel_results['total'] += 1
                per_novel_word[label_a][labels[index]]['count'] += 1
                per_novel_word[label_a][label_a]['encountered'] += 1
                per_novel_word[label_a][label_b]['encountered'] += 1
                t += 1

                scores = attention.one_to_many_score(im, query_b[i, :, :].unsqueeze(0)).squeeze()
                index = torch.argmax(scores).item()

                if index == 1: 
                    novel_results['correct'] += 1
                    c += 1
                novel_results['total'] += 1
                per_novel_word[label_b][labels[index]]['count'] += 1
                per_novel_word[label_b][label_a]['encountered'] += 1
                per_novel_word[label_b][label_b]['encountered'] += 1
                t += 1

        me.append(novel_results)


results = {
    'novel': me
    }
for r in results:
    scores = []
    for d in results[r]:
        # d = results[r]
        c = d['correct']
        t = d['total']
        p = round(100*c/t, 2)
        print(f'{r:<12}: {c}/{t}={p:.2f}%')
        scores.append(p)

    m = np.mean(scores)
    v = np.std(scores)
    print(f'mean: {m:.2f}% std: {v:.2f}%\n')