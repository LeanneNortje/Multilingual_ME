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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--resume", action="store_true", dest="resume",
        help="load from exp_dir if True")
parser.add_argument("--config-file", type=str, default='matchmap', choices=['matchmap'], help="Model config file.")
parser.add_argument("--restore-epoch", type=int, default=-1, help="Epoch to generate accuracies for.")
parser.add_argument("--image-base", default="/storage", help="Model config file.")
command_line_args = parser.parse_args()
restore_epoch = command_line_args.restore_epoch
image_base = command_line_args.image_base
rank = 'cuda'

audio_datapoints = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_datapoints'].item()
image_datapoints = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_datapoints'].item()

query = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_1'].item()
query_tag = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_tag_1'].item()
other_query = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_2'].item()
other_query_tag = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_tag_2'].item()
image_1 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_1'].item()
image_2 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_2'].item()
image_tag_1 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_tag_1'].item()
image_tag_2 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_tag_2'].item()

for args_fn in Path('model_metadata').rglob('*/args.pkl'):
    with open(args_fn, "rb") as f:
        args = pickle.load(f)
    # # Setting up model specifics
    # heading(f'\nSetting up model files ')
    # args, image_base = modelSetup(command_line_args, True)
    

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
    french_model = mutlimodal(args).to(rank)
    dutch_model = mutlimodal(args).to(rank)
    image_model = vision(args).to(rank)

    attention = ScoringAttentionModule(args).to(rank)
    contrastive_loss = ContrastiveLoss(args).to(rank)

    model_with_params_to_update = {
        "enlish_model": english_model,
        "french_model": french_model,
        "dutch_model": dutch_model,
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

    scaler = torch.cuda.amp.GradScaler()

    english_model = DDP(english_model, device_ids=[rank])
    french_model = DDP(french_model, device_ids=[rank])
    dutch_model = DDP(dutch_model, device_ids=[rank])
    image_model = DDP(image_model, device_ids=[rank]) 


    name = 'dutch_english_french_familiar_familiar_trajectory'
    if args['cpc']['warm_start']: name += '_cpc'
    if args['pretrained_alexnet']: name += '_alexnet'

    fn = Path(f'results/{name}.txt')
    done = []

    if fn.is_file(): 
    
        f = open(fn, 'r')
        for line in f:
            num = line.split()[0].split(':')[0]
            done.append(int(num))


    for epoch in range(0, args['n_epochs']+1):
        # if (args["exp_dir"] / Path('models') / Path(f'epoch_{epoch}.pt')).is_file():
        if epoch in done: continue

        if epoch != 0:
            info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAtEpochAMP(
                args["exp_dir"], english_model, image_model, attention, contrastive_loss, optimizer, rank, epoch
                )
            f = open(fn, 'a') 
        else: f = open(fn, 'w') 
        
        with torch.no_grad():
            c = 0
            t = 0

            for ep_num in tqdm(query):


                tags_1 = query_tag[ep_num]
                tags_2 = other_query_tag[ep_num]
                this_image_tags_1 = image_tag_1[ep_num]
                this_image_tags_2 = image_tag_2[ep_num]

                query_output = []
                for i, name in enumerate(query[ep_num]):
                    if tags_1[i] == 'familiar_1':
                        query_output.append(audio_datapoints[name])
                query_output= torch.cat(query_output, dim=0)
                _, _, query_output = english_model(query_output.to(rank))
                
                one = []
                for i, name in enumerate(image_1[ep_num]):
                    if this_image_tags_1[i] == 'familiar_1':
                        one.append(image_datapoints[name])
                one = torch.cat(one, dim=0)
                one = image_model(one.to(rank))

                other_query_output = []
                for i, name in enumerate(other_query[ep_num]):
                    if tags_2[i] == 'familiar_2':
                        other_query_output.append(audio_datapoints[name])
                other_query_output= torch.cat(other_query_output, dim=0)
                _, _, other_query_output = english_model(other_query_output.to(rank))

                two = []
                for i, name in enumerate(image_2[ep_num]):
                    if this_image_tags_2[i] == 'familiar_2':
                        two.append(image_datapoints[name])
                two = torch.cat(two, dim=0)
                two = image_model(two.to(rank))

                for i in range(query_output.size(0)):
                    images = torch.cat([one[i, :, :].unsqueeze(0), two[i, :, :].unsqueeze(0)], dim=0)
                    scores = attention.one_to_many_score(images, query_output[i, :, :].unsqueeze(0)).squeeze()
                    index = torch.argmax(scores).item()
                    if index == 0: c += 1
                    t += 1

                for i in range(other_query_output.size(0)):
                    images = torch.cat([one[i, :, :].unsqueeze(0), two[i, :, :].unsqueeze(0)], dim=0)
                    scores = attention.one_to_many_score(images, other_query_output[i, :, :].unsqueeze(0)).squeeze()
                    index = torch.argmax(scores).item()
                    if index == 1: c += 1
                    t += 1

            print(f'{epoch:<3}: {100*c/t}\n')
            f.write(f'{epoch:<3}: {100*c/t}\n')