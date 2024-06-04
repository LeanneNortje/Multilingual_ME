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
    french_output,
    dutch_output, 
    negatives, positives, attention, contrastive_loss, 
    margin, simtype, alphas, rank):

    i_e = compute_matchmap_similarity_matrix(image_outputs, english_output, attention, simtype)
    i_f = compute_matchmap_similarity_matrix(image_outputs, french_output, attention, simtype)
    i_d = compute_matchmap_similarity_matrix(image_outputs, dutch_output, attention, simtype)

    e_f = compute_matchmap_similarity_matrix(english_output, french_output, attention, simtype)
    e_d = compute_matchmap_similarity_matrix(english_output, dutch_output, attention, simtype)
    f_d = compute_matchmap_similarity_matrix(french_output, dutch_output, attention, simtype)

    neg_i_e = []
    neg_e_i = []
    neg_i_f = []
    neg_f_i = []
    neg_i_d = []
    neg_d_i = []

    neg_e_f = []
    neg_f_e = []
    neg_e_d = []
    neg_d_e = []
    neg_f_d = []
    neg_d_f = []

    for neg_dict in negatives:
        s = compute_matchmap_similarity_matrix(image_outputs, neg_dict["english_output"], attention, simtype)
        neg_i_e.append(s)
        s = compute_matchmap_similarity_matrix(neg_dict['image'], english_output, attention, simtype)
        neg_e_i.append(s)

        s = compute_matchmap_similarity_matrix(image_outputs, neg_dict["french_output"], attention, simtype)
        neg_i_f.append(s)
        s = compute_matchmap_similarity_matrix(neg_dict['image'], french_output, attention, simtype)
        neg_f_i.append(s)

        s = compute_matchmap_similarity_matrix(image_outputs, neg_dict["dutch_output"], attention, simtype)
        neg_i_d.append(s)
        s = compute_matchmap_similarity_matrix(neg_dict['image'], dutch_output, attention, simtype)
        neg_d_i.append(s)



        s = compute_matchmap_similarity_matrix(english_output, neg_dict["french_output"], attention, simtype)
        neg_e_f.append(s)
        s = compute_matchmap_similarity_matrix(french_output, neg_dict["english_output"], attention, simtype)
        neg_f_e.append(s)

        s = compute_matchmap_similarity_matrix(english_output, neg_dict["dutch_output"], attention, simtype)
        neg_e_d.append(s)
        s = compute_matchmap_similarity_matrix(dutch_output, neg_dict["english_output"], attention, simtype)
        neg_d_e.append(s)

        s = compute_matchmap_similarity_matrix(french_output, neg_dict["dutch_output"], attention, simtype)
        neg_f_d.append(s)
        s = compute_matchmap_similarity_matrix(dutch_output, neg_dict["french_output"], attention, simtype)
        neg_d_f.append(s)

    neg_i_e = torch.cat(neg_i_e, dim=1)
    neg_e_i = torch.cat(neg_e_i, dim=1)
    neg_i_f = torch.cat(neg_i_f, dim=1)
    neg_f_i = torch.cat(neg_f_i, dim=1)
    neg_i_d = torch.cat(neg_i_d, dim=1)
    neg_d_i = torch.cat(neg_d_i, dim=1)

    neg_e_f = torch.cat(neg_e_f, dim=1)
    neg_f_e = torch.cat(neg_f_e, dim=1)
    neg_e_d = torch.cat(neg_e_d, dim=1)
    neg_d_e = torch.cat(neg_d_e, dim=1)
    neg_f_d = torch.cat(neg_f_d, dim=1)
    neg_d_f = torch.cat(neg_d_f, dim=1)



    pos_i_e = []
    pos_e_i = []
    pos_i_f = []
    pos_f_i = []
    pos_i_d = []
    pos_d_i = []

    pos_e_f = []
    pos_f_e = []
    pos_e_d = []
    pos_d_e = []
    pos_f_d = []
    pos_d_f = []

    for pos_dict in positives:
        s = compute_matchmap_similarity_matrix(image_outputs, pos_dict["english_output"], attention, simtype)
        pos_i_e.append(s)
        s = compute_matchmap_similarity_matrix(pos_dict['image'], english_output, attention, simtype)
        pos_e_i.append(s)

        s = compute_matchmap_similarity_matrix(image_outputs, pos_dict["french_output"], attention, simtype)
        pos_i_f.append(s)
        s = compute_matchmap_similarity_matrix(pos_dict['image'], french_output, attention, simtype)
        pos_f_i.append(s)

        s = compute_matchmap_similarity_matrix(image_outputs, pos_dict["dutch_output"], attention, simtype)
        pos_i_d.append(s)
        s = compute_matchmap_similarity_matrix(pos_dict['image'], dutch_output, attention, simtype)
        pos_d_i.append(s)


        s = compute_matchmap_similarity_matrix(english_output, pos_dict["french_output"], attention, simtype)
        pos_e_f.append(s)
        s = compute_matchmap_similarity_matrix(french_output, pos_dict["english_output"], attention, simtype)
        pos_f_e.append(s)

        s = compute_matchmap_similarity_matrix(english_output, pos_dict["dutch_output"], attention, simtype)
        pos_e_d.append(s)
        s = compute_matchmap_similarity_matrix(dutch_output, pos_dict["english_output"], attention, simtype)
        pos_d_e.append(s)

        s = compute_matchmap_similarity_matrix(french_output, pos_dict["dutch_output"], attention, simtype)
        pos_f_d.append(s)
        s = compute_matchmap_similarity_matrix(dutch_output, pos_dict["french_output"], attention, simtype)
        pos_d_f.append(s)

    pos_i_e = torch.cat(pos_i_e, dim=1)
    pos_e_i = torch.cat(pos_e_i, dim=1)
    pos_i_f = torch.cat(pos_i_f, dim=1)
    pos_f_i = torch.cat(pos_f_i, dim=1)
    pos_i_d = torch.cat(pos_i_d, dim=1)
    pos_d_i = torch.cat(pos_d_i, dim=1)

    pos_e_f = torch.cat(pos_e_f, dim=1)
    pos_f_e = torch.cat(pos_f_e, dim=1)
    pos_e_d = torch.cat(pos_e_d, dim=1)
    pos_d_e = torch.cat(pos_d_e, dim=1)
    pos_f_d = torch.cat(pos_f_d, dim=1)
    pos_d_f = torch.cat(pos_d_f, dim=1)

    loss = contrastive_loss(i_e.unsqueeze(-1), pos_i_e.unsqueeze(-1), pos_e_i.unsqueeze(-1), neg_i_e.unsqueeze(-1), neg_e_i.unsqueeze(-1))
    loss += contrastive_loss(i_f.unsqueeze(-1), pos_i_f.unsqueeze(-1), pos_f_i.unsqueeze(-1), neg_i_f.unsqueeze(-1), neg_f_i.unsqueeze(-1))
    loss += contrastive_loss(i_d.unsqueeze(-1), pos_i_d.unsqueeze(-1), pos_d_i.unsqueeze(-1), neg_i_d.unsqueeze(-1), neg_d_i.unsqueeze(-1))

    loss += contrastive_loss(e_f.unsqueeze(-1), pos_e_f.unsqueeze(-1), pos_f_e.unsqueeze(-1), neg_e_f.unsqueeze(-1), neg_f_e.unsqueeze(-1))  
    loss += contrastive_loss(e_d.unsqueeze(-1), pos_e_d.unsqueeze(-1), pos_d_e.unsqueeze(-1), neg_e_d.unsqueeze(-1), neg_d_e.unsqueeze(-1))  
    loss += contrastive_loss(f_d.unsqueeze(-1), pos_f_d.unsqueeze(-1), pos_d_f.unsqueeze(-1), neg_f_d.unsqueeze(-1), neg_d_f.unsqueeze(-1))

    return loss