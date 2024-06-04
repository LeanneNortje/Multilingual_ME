#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from itertools import product
from tqdm import tqdm
from models.util import *
from tqdm import tqdm

def compute_matchmap_similarity_matrix(a, b, attention, simtype='MISA'):
    
    assert(a.dim() == 3)
    assert(b.dim() == 3)
    
    n = a.size(0)
    scores = []

    for i in range(n):
        s = attention(a[i, :, :].unsqueeze(0), b[i, :, :].unsqueeze(0))
        scores.append(s)

    scores = torch.cat(scores, dim=0) 

    return scores

def compute_matchmap_similarity_matrix_IA(im, im_mask, audio, frames, attention, simtype='MISA'):
    
    w = im.size(2)
    h = im.size(3)
    im = im.view(im.size(0), im.size(1), -1).transpose(1, 2)
    audio = audio.squeeze(2)

    assert(im.dim() == 3)
    assert(audio.dim() == 3)
    
    n = im.size(0)
    scores = []

    for i in range(n):
        s = attention(im[i, :, :].unsqueeze(0), audio[i, :, :].unsqueeze(0), frames[i])
        scores.append(s)

    scores = torch.cat(scores, dim=0) 

    return scores
    
def compute_large_matchmap_similarity_matrix_IA(im, im_mask, audio, frames, attention, simtype='MISA'):
    
    w = im.size(2)
    h = im.size(3)
    im = im.view(im.size(0), im.size(1), -1).transpose(1, 2)
    audio = audio.squeeze(2)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    assert(im.dim() == 3)
    assert(audio.dim() == 3)
    
    n = im.size(0)
    S = []
    im_C = []

    for i in range(n):
        this_im_row = []
        for j in range(n):
            this_aud, this_im = attention(im[i, :, :].unsqueeze(0), audio[j, :, :].unsqueeze(0), frames[j])
            score = cos(this_aud, this_im)
            this_im_row.append(score.unsqueeze(1))
        this_im_row = torch.cat(this_im_row, dim=1)
        S.append(this_im_row)

    S = torch.cat(S, dim=0) 

    return S

def compute_matchmap_similarity_score_IA(im, im_mask, audio, frames, attention, simtype='MISA'):
    
    # w = im.size(2)
    # h = im.size(3)
    # im = im.view(im.size(0), im.size(1), -1).transpose(1, 2)
    # audio = audio.squeeze(2)

    assert(im.dim() == 3)
    assert(audio.dim() == 3)
    
    S, C = attention(im, audio, frames)
    C = C.unsqueeze(0)

    return S, C