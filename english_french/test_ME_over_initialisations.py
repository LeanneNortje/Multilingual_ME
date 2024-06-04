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

rewind = "\033[A"*2

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

episodes = np.load(Path('data/episodes.npz'), allow_pickle=True)['episodes'].item()
model_results = {
    'familiar_1': {}, 
    'familiar_2': {},
    'novel': {}, 
    'known_novel': {}
    }

for args_fn in Path('model_metadata').rglob('*/args.pkl'):
    with open(args_fn, "rb") as f:
        args = pickle.load(f)
    
    one_familiar_results = {'correct': 0, 'total': 0}
    two_familiar_results = {'correct': 0, 'total': 0}
    novel_results = {'correct': 0, 'total': 0}
    known_novel_results = {'correct': 0, 'total': 0}

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
    dutch_model = mutlimodal(args).to(rank)
    french_model = mutlimodal(args).to(rank)
    image_model = vision(args).to(rank)

    attention = ScoringAttentionModule(args).to(rank)
    contrastive_loss = ContrastiveLoss(args).to(rank)

    model_with_params_to_update = {
        "english_model": english_model,
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
    # attention = DDP(attention, device_ids=[rank])


    if "restore_epoch" in args:
        info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAtEpochAMP(
            args["exp_dir"], english_model, french_model, dutch_model, image_model, attention, contrastive_loss, optimizer, rank, 
            args["restore_epoch"]
            )
    else: 
        heading(f'\nRetoring model parameters from best epoch ')
        info, epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAMP(
            args["exp_dir"], english_model, french_model, dutch_model, image_model, attention, contrastive_loss, optimizer, 
            rank, False
            )

    c = 0
    t = 0
    with torch.no_grad():

        for ep_num in tqdm(episodes, desc=f'{rewind}'):
        # if ep_num != 25: continue
            episode = episodes[ep_num]


            query_output = []
            for name in query[ep_num]:
                # print(name)
                query_output.append(audio_datapoints[name])
            query_output= torch.cat(query_output, dim=0)
            _, _, query_output = english_model(query_output.to(rank))
            # break

            query_image = []
            for name in image_1[ep_num]:
                query_image.append(image_datapoints[name])
            query_image = torch.cat(query_image, dim=0)
            query_image = image_model(query_image.to(rank))


            secondary_query_output = []
            for name in secondary_query[ep_num]:
                secondary_query_output.append(audio_datapoints[name])
            secondary_query_output= torch.cat(secondary_query_output, dim=0)
            _, _, secondary_query_output = english_model(secondary_query_output.to(rank))

            other_image = []
            for name in image_2[ep_num]:
                other_image.append(image_datapoints[name])
            other_image = torch.cat(other_image, dim=0)
            other_image = image_model(other_image.to(rank))


            tags_1 = query_tag[ep_num]
            labels_1 = query_labels[ep_num]
            tags_2 = secondary_tag[ep_num]
            labels_2 = secondary_labels[ep_num]

       
            for i in range(query_output.size(0)):

                images = torch.cat([query_image[i, :, :].unsqueeze(0), other_image[i, :, :].unsqueeze(0)], dim=0)
        

                scores = attention.one_to_many_score(images, query_output[i, :, :].unsqueeze(0)).squeeze()
                index = torch.argmax(scores).item()



                if tags_1[i] == 'familiar_1':
                    if index == 0: 
                        one_familiar_results['correct'] += 1
                    one_familiar_results['total'] += 1

                elif tags_1[i] == 'novel':
                    if index == 0: 
                        novel_results['correct'] += 1
                        c += 1
                    novel_results['total'] += 1
                    t += 1

                scores = attention.one_to_many_score(images, secondary_query_output[i, :, :].unsqueeze(0)).squeeze()
                index = torch.argmax(scores).item()

                if tags_2[i] == 'familiar_2':
                    if index == 1: 
                        two_familiar_results['correct'] += 1
                    two_familiar_results['total'] += 1
                elif tags_2[i] == 'known_novel':
                    if index == 1: known_novel_results['correct'] += 1
                    known_novel_results['total'] += 1
            print(f'{c}/{t}={100*c/t}%')

            # if ep_num == 9: break
            # break

    instance = args_fn.parent.stem
    if instance not in  model_results['familiar_1']: model_results['familiar_1'][instance] = one_familiar_results
    if instance not in  model_results['familiar_2']: model_results['familiar_2'][instance] = two_familiar_results
    if instance not in  model_results['novel']: model_results['novel'][instance] = novel_results
    if instance not in  model_results['known_novel']: model_results['known_novel'][instance] = known_novel_results


for test in model_results:

    c = 0
    t = 0
    for name in model_results[test]:
        d = model_results[test][name]
        c += d['correct']
        t += d['total']
    p = round(100*c/t, 2)
    print(f'{test:<12}: {c}/{t}={p:.2f}%')

    # name = 'keyword'
    # if args['cpc']['warm_start']: name += '_cpc'
    # if args['pretrained_alexnet']: name += '_alexnet'
    # name += f'_{args["instance"]}'

    # f = open(Path(f'results/files/{name}.txt'), 'w')

    # for w in per_novel_word:

    #     d = per_novel_word[w]
    #     c = d['correct']
    #     t = d['total']
    #     p = round(100*c/t, 2)
    #     print(f'{w:<12}: {c}/{t}={p:.2f}%')
    #     f.write(f'{w:<12} {p}\n')