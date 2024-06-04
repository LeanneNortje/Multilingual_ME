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
    anchors, negatives, positives, attention, contrastive_loss, 
    args, margin, simtype, rank):
    # loss = 0

    loss_terms = {}

    if args['training_languages']['english']:
        loss_terms['english'] = {'anch_image_pos_audio': [], 'anch_audio_pos_image': [], 'anch_image_neg_audio': [], 'anch_audio_neg_image': []}
        loss_terms['english']['anch_image_anch_audio'] = compute_matchmap_similarity_matrix(anchors['image_outputs'], anchors['english_output'], attention, simtype)

    if args['training_languages']['dutch']:
        loss_terms['dutch'] = {'anch_image_pos_audio': [], 'anch_audio_pos_image': [], 'anch_image_neg_audio': [], 'anch_audio_neg_image': []}
        loss_terms['dutch']['anch_image_anch_audio'] = compute_matchmap_similarity_matrix(anchors['image_outputs'], anchors['dutch_output'], attention, simtype)

    if args['training_languages']['french']:
        loss_terms['french'] = {'anch_image_pos_audio': [], 'anch_audio_pos_image': [], 'anch_image_neg_audio': [], 'anch_audio_neg_image': []}
        loss_terms['french']['anch_image_anch_audio'] = compute_matchmap_similarity_matrix(anchors['image_outputs'], anchors['french_output'], attention, simtype)


    for neg_dict in negatives:
        if args['training_languages']['english']:
            s = compute_matchmap_similarity_matrix(anchors['image_outputs'], neg_dict["english_output"], attention, simtype)
            loss_terms['english']['anch_image_neg_audio'].append(s)
            s = compute_matchmap_similarity_matrix(neg_dict['image'], anchors['english_output'], attention, simtype)
            loss_terms['english']['anch_audio_neg_image'].append(s)

        if args['training_languages']['dutch']:
            s = compute_matchmap_similarity_matrix(anchors['image_outputs'], neg_dict["dutch_output"], attention, simtype)
            loss_terms['dutch']['anch_image_neg_audio'].append(s)
            s = compute_matchmap_similarity_matrix(neg_dict['image'], anchors['dutch_output'], attention, simtype)
            loss_terms['dutch']['anch_audio_neg_image'].append(s)

        if args['training_languages']['french']:
            s = compute_matchmap_similarity_matrix(anchors['image_outputs'], neg_dict["french_output"], attention, simtype)
            loss_terms['french']['anch_image_neg_audio'].append(s)
            s = compute_matchmap_similarity_matrix(neg_dict['image'], anchors['french_output'], attention, simtype)
            loss_terms['french']['anch_audio_neg_image'].append(s)


    for pos_dict in positives:
        if args['training_languages']['english']:
            s = compute_matchmap_similarity_matrix(anchors['image_outputs'], pos_dict["english_output"], attention, simtype)
            loss_terms['english']['anch_image_pos_audio'].append(s)
            s = compute_matchmap_similarity_matrix(pos_dict['image'], anchors['english_output'], attention, simtype)
            loss_terms['english']['anch_audio_pos_image'].append(s)

        if args['training_languages']['dutch']:
            s = compute_matchmap_similarity_matrix(anchors['image_outputs'], pos_dict["dutch_output"], attention, simtype)
            loss_terms['dutch']['anch_image_pos_audio'].append(s)
            s = compute_matchmap_similarity_matrix(pos_dict['image'], anchors['dutch_output'], attention, simtype)
            loss_terms['dutch']['anch_audio_pos_image'].append(s)

        if args['training_languages']['french']:
            s = compute_matchmap_similarity_matrix(anchors['image_outputs'], pos_dict["french_output"], attention, simtype)
            loss_terms['french']['anch_image_pos_audio'].append(s)
            s = compute_matchmap_similarity_matrix(pos_dict['image'], anchors['french_output'], attention, simtype)
            loss_terms['french']['anch_audio_pos_image'].append(s)

    loss = 0
    for language in loss_terms:
        for key in loss_terms[language]:
            if key != 'anch_image_anch_audio': loss_terms[language][key] = torch.cat(loss_terms[language][key], dim=1)

        loss += contrastive_loss(
            loss_terms[language]['anch_image_anch_audio'].unsqueeze(-1), 
            loss_terms[language]['anch_image_pos_audio'].unsqueeze(-1), 
            loss_terms[language]['anch_audio_pos_image'].unsqueeze(-1), 
            loss_terms[language]['anch_image_neg_audio'].unsqueeze(-1), 
            loss_terms[language]['anch_audio_neg_image'].unsqueeze(-1)
            )  

    return loss