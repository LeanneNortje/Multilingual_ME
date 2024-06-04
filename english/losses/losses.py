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
from .util import *
    
def compute_matchmap_similarity_matrix_loss(
    image_outputs, english_output, 
    negatives, positives, attention, contrastive_loss, 
    margin, simtype, alphas, rank):
    # print(image_outputs.size(), english_output.size())
    i_e = compute_matchmap_similarity_matrix(image_outputs, english_output, attention, simtype)
    # print(i_e.size())
    neg_i_e = []
    neg_e_i = []

    for neg_dict in negatives:
        s = compute_matchmap_similarity_matrix(image_outputs, neg_dict["english_output"], attention, simtype)
        neg_i_e.append(s)
        s = compute_matchmap_similarity_matrix(neg_dict['image'], english_output, attention, simtype)
        neg_e_i.append(s)

    neg_i_e = torch.cat(neg_i_e, dim=1)
    neg_e_i = torch.cat(neg_e_i, dim=1)

    pos_i_e = []
    pos_e_i = []

    for pos_dict in positives:
        s = compute_matchmap_similarity_matrix(image_outputs, pos_dict["english_output"], attention, simtype)
        pos_i_e.append(s)
        s = compute_matchmap_similarity_matrix(pos_dict['image'], english_output, attention, simtype)
        pos_e_i.append(s)


    pos_i_e = torch.cat(pos_i_e, dim=1)
    pos_e_i = torch.cat(pos_e_i, dim=1)

    loss = contrastive_loss(i_e.unsqueeze(-1), pos_i_e.unsqueeze(-1), pos_e_i.unsqueeze(-1), neg_i_e.unsqueeze(-1), neg_e_i.unsqueeze(-1))  

    return loss