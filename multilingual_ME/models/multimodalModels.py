#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
from torch import Tensor
import numpy as np
import math
from torchvision.io import read_image
from torchvision.models import *

class mutlimodal(nn.Module):
    def __init__(self, args):
        super(mutlimodal, self).__init__()
        num_channels = args["acoustic_model"]["out_channels"]
        z_dim = args["audio_model"]["z_dim"]
        c_dim = args["audio_model"]["c_dim"]
        frame_dim = args["audio_config"]["target_length"]

        self.conv = nn.Conv1d(
            args["acoustic_model"]["in_channels"], num_channels, 
            args["acoustic_model"]["kernel_size"], 
            args["acoustic_model"]["stride"], 
            args["acoustic_model"]["padding"], bias=False)

        self.encoder = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # nn.InstanceNorm1d(num_channels),

            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # nn.InstanceNorm1d(num_channels),

            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # nn.InstanceNorm1d(num_channels),

            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # nn.InstanceNorm1d(num_channels),

            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, z_dim),
            # nn.InstanceNorm1d(z_dim),
        )
        self.rnn1 = nn.LSTM(z_dim, c_dim, batch_first=True)
        self.rnn2 = nn.LSTM(c_dim, c_dim, batch_first=True)
        self.rnn3 = nn.LSTM(c_dim, c_dim, batch_first=True)
        self.rnn4 = nn.LSTM(c_dim, c_dim, batch_first=True)

        self.english_rnn1 = nn.LSTM(512, 512, batch_first=True, bidirectional=True)
        self.english_rnn2 = nn.LSTM(1024, 1024, batch_first=True, bidirectional=True)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        # self.ln = nn.LayerNorm(512)
        self.audio_encoder = nn.Sequential(
            nn.Linear(frame_dim//2, frame_dim//4),
            # nn.ReLU(),
            # nn.Linear(frame_dim//4, frame_dim//8),
            # # nn.ReLU(),
            # nn.Linear(frame_dim//8, frame_dim//16),
            nn.LeakyReLU(),
            nn.Linear(frame_dim//4, 1),
            nn.LeakyReLU()
        )

    def forward(self, mels):
        z = self.conv(mels)
        z = self.relu(z)
        z = self.encoder(z.transpose(1, 2))

        c, _ = self.rnn1(z)
        c, _ = self.rnn2(c)
        c, _ = self.rnn3(c)
        c, _ = self.rnn4(c)

        s, _ = self.english_rnn1(c)
        # s = self.relu(s)
        s, _ = self.english_rnn2(s)
        # s = self.relu(s)
        s = s.transpose(1, 2)
        s = self.audio_encoder(s)

        return z, z, s

class vision(nn.Module):
    def __init__(self, args):
        super(vision, self).__init__()
        model_weights = torch.load("pretrained/ckpt.pth")
        seed_model = alexnet(pretrained=False)
        self.image_model = nn.Sequential(*list(seed_model.features.children()))

        last_layer_index = len(list(self.image_model.children()))
        self.image_model.add_module(str(last_layer_index),
            nn.Conv2d(256, args["audio_model"]["embedding_dim"], kernel_size=(3,3), stride=(1,1), padding=(1,1)))
        
        for name, param in self.image_model.named_parameters():
            print(name, param[0][0])
            break
        if args['pretrained_alexnet']:
            print('Using pretrained AlexNet')
            model_dict = self.image_model.state_dict()
            for key in model_weights['model']:
                if 'features' in key: 
                    new_key = '.'.join(key.split('.')[2:])
                    model_dict[new_key] = model_weights['model'][key]
            self.image_model.load_state_dict(model_dict)

        for name, param in self.image_model.named_parameters():
            print(name, param[0][0])
            break
            
    def forward(self, x):

        x = self.image_model(x)
        x = x.view(x.size(0), x.size(1), -1)
        # x = self.image_encoder(x)
        return x