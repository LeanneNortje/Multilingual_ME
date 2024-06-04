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
image_1 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_1'].item()
image_2 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_2'].item()
image_tag_1 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_tag_1'].item()
image_tag_2 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_tag_2'].item()

heading(f'\nSetting up model files ')
args, image_base = modelSetup(command_line_args, True)
    
name = 'baseline_trajectory'

f = open(Path(f'results/{name}.txt'), 'w')


for epoch in range(0, args['n_epochs']+1):

    with torch.no_grad():
        c = 0
        t = 0

        for ep_num in tqdm(query):

            tags_1 = query_tag[ep_num]
            this_image_tags_1 = image_tag_1[ep_num]
            this_image_tags_2 = image_tag_2[ep_num]


            for i in range(len(tags_1)):
                index = np.random.randint(0, 2)
                if index == 0: c += 1
                t += 1

        print(f'{epoch:<3}: {100*c/t}\n')
        f.write(f'{epoch:<3}: {100*c/t}\n')