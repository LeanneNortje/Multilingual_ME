#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
# adapted from https://github.com/dharwath

import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from losses import compute_matchmap_similarity_matrix_loss
from dataloaders import *
from models.setup import *
from models.util import *
from models.multimodalModels import *
from models.GeneralModels import *
from models.infonce import infonce
from evaluation.calculations import *
from training.util import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import time
from torchvision.io import read_image
from torchvision.models import *

BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"

def collapse_pixels(im):
    im = im.view(im.size(0), im.size(1), -1).transpose(1, 2)
    return  im#.mean(dim=1)

def loadImage(impath, args):
    image_conf = args["image_config"]
    crop_size = image_conf.get('crop_size')
    center_crop = image_conf.get('center_crop')
    RGB_mean = image_conf.get('RGB_mean')
    RGB_std = image_conf.get('RGB_std')

    resize = transforms.Resize((256, 256))
    to_tensor = transforms.ToTensor()
    image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

    img = Image.open(impath).convert('RGB')
    img = resize(img)
    img = to_tensor(img)
    img = image_normalize(img)
    return img

def validate(audio_model, image_model, attention, contrastive_loss, val_loader, rank, image_base, args):
    # function adapted from https://github.com/dharwath
    start_time = time.time()
    N_examples = val_loader.dataset.__len__()
    I_embeddings = [] 
    anch = []
    # positives = []
    negatives = []
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    with torch.no_grad():

        for value_dict in tqdm(val_loader, leave=False):

            image_output = image_model(value_dict["image"].to(rank))
            
            english_input = value_dict["english_feat"].to(rank)
            _, _, english_output = audio_model(english_input)
            english_nframes = NFrames(english_input, english_output, value_dict["english_nframes"]) 
            anch_s = attention.score(image_output, english_output)
            anch.append(anch_s)

            # ps = []
            # for p, pos_dict in enumerate(value_dict['positives']):
            #     pos_image_output = image_model(pos_dict["pos_image"].to(rank))
            #     score = attention.score(pos_image_output, english_output)
            #     ps.append(score)
            # positives.append(torch.cat(ps, dim=1))

            ns = []
            for neg_dict in value_dict['negatives']:
                neg_image_output = image_model(neg_dict['neg_image'].to(rank))
                score = attention.score(neg_image_output, english_output)
                ns.append(score)
            negatives.append(torch.cat(ns, dim=1))

        anch = torch.cat(anch, dim=0)
        # positives = torch.cat(positives, dim=0)
        negatives = torch.cat(negatives, dim=0)
        scores = torch.cat([anch, negatives], dim=1)
        treshold = anch.size(1) #+ positives.size(1)
        ind = torch.argsort(scores, dim=1, descending=True)[:, 0:treshold]
        acc = (ind[:, 0:treshold] <= treshold).float().mean().detach().item()

        end_time = time.time()
        days, hours, minutes, seconds = timeFormat(start_time, end_time)
        print(f'Prediction accuracy: {acc*100}%')
        print(f'Validation took {hours:>2} hours {minutes:>2} minutes {seconds:>2} seconds')

    return acc
    
def spawn_training(rank, world_size, image_base, args):

    # # Create dataloaders
    dist.init_process_group(
        BACKEND,
        rank=rank,
        world_size=world_size,
        init_method=INIT_METHOD,
    )
    torch.manual_seed(42)

    if rank == 0: writer = SummaryWriter(args["exp_dir"] / "tensorboard")
    best_epoch, best_acc = 0, 0
    global_step, start_epoch = 0, 0
    info = {}
    loss_tracker = valueTracking()

    if rank == 0: heading(f'\nLoading training data ')
    train_dataset = ImageAudioDatawithSampling(image_base, args["data_train"], Path("data/train_lookup.npz"), args, rank)
    train_sampler = DistributedSampler(train_dataset, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )


    if rank == 0: 
        heading(f'\nLoading validation data ')
        args["image_config"]["center_crop"] = True
        validation_loader = torch.utils.data.DataLoader(
            ImageAudioDatawithSamplingVal(image_base, args["data_val"], Path("data/val_lookup.npz"), args, rank),
            batch_size=args["batch_size"], shuffle=False, num_workers=1, pin_memory=True)
    
    if rank == 0: heading(f'\nSetting up Audio model ')
    audio_model = mutlimodal(args).to(rank)

    if rank == 0: heading(f'\nSetting up image model ')
    image_model = vision(args).to(rank)

    if rank == 0: heading(f'\nSetting up attention model ')
    attention = ScoringAttentionModule(args).to(rank)

    if rank == 0: heading(f'\nSetting up contrastive loss ')
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

    if args["resume"] is False and args['cpc']['warm_start']: 
        if rank == 0: print("Loading pretrained acoustic weights")
        audio_model = loadPretrainedWeights(audio_model, args, rank)
 
    if args["resume"]:

        if "restore_epoch" in args:
            info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAtEpochAMP(
                args["exp_dir"], audio_model, image_model, attention, contrastive_loss, optimizer, rank, args["restore_epoch"]
                )
            if rank == 0: print(f'\nEpoch particulars:\n\t\tepoch = {start_epoch}\n\t\tglobal_step = {global_step}\n\t\tbest_epoch = {best_epoch}\n\t\tbest_acc = {best_acc}\n')
        else:
            info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAMP(
                args["exp_dir"], audio_model, image_model, attention, contrastive_loss, optimizer, rank
                )
            if rank == 0: print(f'\nEpoch particulars:\n\t\tepoch = {start_epoch}\n\t\tglobal_step = {global_step}\n\t\tbest_epoch = {best_epoch}\n\t\tbest_acc = {best_acc}\n')

    start_epoch += 1

    for epoch in np.arange(start_epoch, args["n_epochs"] + 1):
        train_sampler.set_epoch(int(epoch))
        current_learning_rate = adjust_learning_rate(args, optimizer, epoch, 0.00001)

        audio_model.train()
        image_model.train()
        attention.train()
        # contrastive_loss.train()

        loss_tracker.new_epoch()
        start_time = time.time()
        if rank == 0: printEpoch(epoch, 0, len(train_loader), loss_tracker, best_acc, start_time, start_time, current_learning_rate)
        
        i = 0
                    
        for value_dict in train_loader:
            
            optimizer.zero_grad() 

            with torch.cuda.amp.autocast():
        
                image_output = image_model(value_dict['image'].to(rank))
                anchors = {'image_outputs': image_output}
                # print('image ', image_output.size())
                
                if args['training_languages']['english']:
                    english_input = value_dict["english_feat"].to(rank)
                    _, _, english_output = audio_model(english_input)
                    anchors['english_output'] = english_output
                    # print('english ', english_output.size())

                if args['training_languages']['dutch']:
                    dutch_input = value_dict["dutch_feat"].to(rank)
                    _, _, dutch_output = audio_model(dutch_input)
                    anchors['dutch_output'] = dutch_output
                    # print('dutch ', dutch_output.size())

                if args['training_languages']['french']:
                    french_input = value_dict['french_feat'].to(rank)
                    _, _, french_output = audio_model(french_input)
                    anchors['french_output'] = french_output
                    # print('french ', french_output.size())
                # print(
                #     torch.all(torch.eq(anchors['french_output'], anchors['english_output'])).item(),
                #     torch.all(torch.eq(anchors['english_output'], anchors['dutch_output'])).item(),
                #     torch.all(torch.eq(anchors['dutch_output'], anchors['french_output'])).item()
                #     )
                    
                positives = []
                for p, pos_dict in enumerate(value_dict['positives']):
                    pos_image_output = image_model(pos_dict["pos_image"].to(rank))#pos_images[p]
                    samples = {"image": pos_image_output}
                    # print(p, ' pos image ', torch.all(torch.eq(pos_image_output, anchors['image_outputs'])).item())


                    if args['training_languages']['english']:
                        pos_english_input = pos_dict['pos_english'].to(rank)
                        _, _, pos_english_output = audio_model(pos_english_input)
                        samples["english_output"] = pos_english_output
                        # print(p, ' pos english ', 
                        #     torch.all(torch.eq(pos_english_output, anchors['english_output'])).item(),
                        #     torch.all(torch.eq(pos_english_output, anchors['dutch_output'])).item(),
                        #     torch.all(torch.eq(pos_english_output, anchors['french_output'])).item()
                        #     )

                    if args['training_languages']['dutch']:
                        pos_dutch_input = pos_dict['pos_dutch'].to(rank)
                        _, _, pos_dutch_output = audio_model(pos_dutch_input)
                        samples['dutch_output'] = pos_dutch_output
                        # print(p, ' pos dutch ',
                        #     torch.all(torch.eq(pos_dutch_output, anchors['english_output'])).item(),
                        #     torch.all(torch.eq(pos_dutch_output, anchors['dutch_output'])).item(),
                        #     torch.all(torch.eq(pos_dutch_output, anchors['french_output'])).item()
                        #     )

                    if args['training_languages']['french']:
                        pos_french_input = pos_dict['pos_french'].to(rank)
                        _, _, pos_french_output = audio_model(pos_french_input)
                        samples['french_output'] = pos_french_output
                        # print(p, ' pos french ', 
                        #     torch.all(torch.eq(pos_french_output, anchors['english_output'])).item(),
                        #     torch.all(torch.eq(pos_french_output, anchors['dutch_output'])).item(),
                        #     torch.all(torch.eq(pos_french_output, anchors['french_output'])).item()
                        #     )
                   
                    positives.append(samples)

                negatives = []
                for n, neg_dict in enumerate(value_dict['negatives']):
                    neg_image_output = image_model(neg_dict['neg_image'].to(rank))#neg_images[n]
                    samples = {"image": neg_image_output}
                    # print(n, ' neg image ', torch.all(torch.eq(neg_image_output, anchors['image_outputs'])).item())

                    if args['training_languages']['english']:
                        neg_english_input = neg_dict['neg_english'].to(rank)
                        _, _, neg_english_output = audio_model(neg_english_input)
                        samples["english_output"] = neg_english_output
                        # print(n, ' neg english ', 
                        #     torch.all(torch.eq(neg_english_output, anchors['english_output'])).item(),
                        #     torch.all(torch.eq(neg_english_output, anchors['dutch_output'])).item(),
                        #     torch.all(torch.eq(neg_english_output, anchors['french_output'])).item()
                        #     )
                        
                    if args['training_languages']['dutch']:
                        neg_dutch_input = neg_dict['neg_dutch'].to(rank)
                        _, _, neg_dutch_output = audio_model(neg_dutch_input)
                        samples['dutch_output'] = neg_dutch_output
                        # print(n, ' neg dutch ', 
                        #     torch.all(torch.eq(neg_dutch_output, anchors['english_output'])).item(),
                        #     torch.all(torch.eq(neg_dutch_output, anchors['dutch_output'])).item(),
                        #     torch.all(torch.eq(neg_dutch_output, anchors['french_output'])).item()
                        #     )

                    if args['training_languages']['french']:
                        neg_french_input = neg_dict['neg_french'].to(rank)
                        _, _, neg_french_output = audio_model(neg_french_input)
                        samples['french_output'] = neg_french_output
                    #     print(n, ' neg french ', 
                    #         torch.all(torch.eq(neg_french_output, anchors['english_output'])).item(),
                    #         torch.all(torch.eq(neg_french_output, anchors['dutch_output'])).item(),
                    #         torch.all(torch.eq(neg_french_output, anchors['french_output'])).item()
                    #         )


                    negatives.append(samples)

                # e_key = 'english_output'
                # d_key =  'dutch_output' 
                # f_key = 'french_output'

                # for d1 in positives:
                    
                #     for d2 in negatives:
                #         print(e_key, 
                #             torch.all(torch.eq(d1[e_key], d2[e_key])).item(),
                #             torch.all(torch.eq(d1[e_key], d2[d_key])).item(),
                #             torch.all(torch.eq(d1[e_key], d2[f_key])).item()
                #             )
                #         print(e_key, 
                #             torch.all(torch.eq(d1[d_key], d2[e_key])).item(),
                #             torch.all(torch.eq(d1[d_key], d2[d_key])).item(),
                #             torch.all(torch.eq(d1[d_key], d2[f_key])).item()
                #             )
                #         print(e_key, 
                #             torch.all(torch.eq(d1[f_key], d2[e_key])).item(),
                #             torch.all(torch.eq(d1[f_key], d2[d_key])).item(),
                #             torch.all(torch.eq(d1[f_key], d2[f_key])).item()
                #             )

                loss = compute_matchmap_similarity_matrix_loss(
                    anchors, negatives, positives, attention, contrastive_loss, args,  
                    margin=args["margin"], simtype=args["simtype"], rank=rank
                    )

            loss.backward()
            optimizer.step()

            loss_tracker.update(loss.detach().item(), anchors['image_outputs'].detach().size(0)) #####
            end_time = time.time()
            if rank == 0: printEpoch(epoch, i+1, len(train_loader), loss_tracker, best_acc, start_time, end_time, current_learning_rate)
            if np.isnan(loss_tracker.average):
                print("training diverged...")
                return
            # else:
            global_step += 1 
            # if i == 10: break
            # break
            i += 1
        # break
        if rank == 0:
            # avg_acc = validate_contrastive(audio_model, image_model, attention, contrastive_loss, validation_loader, rank, args)
            avg_acc = validate(audio_model, image_model, attention, contrastive_loss, validation_loader, rank, image_base, args)
            # break
            writer.add_scalar("loss/train", loss_tracker.average, epoch)
            writer.add_scalar("loss/val", avg_acc, epoch)

            best_acc, best_epoch = saveModelAttriburesAndTrainingAMP(
                args["exp_dir"], audio_model, 
                image_model, attention, contrastive_loss, optimizer, info, int(epoch), global_step, best_epoch, avg_acc, best_acc, loss_tracker.average, end_time-start_time)
        # break
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume", action="store_true", dest="resume",
            help="load from exp_dir if True")
    parser.add_argument("--config-file", type=str, default='matchmap', choices=['matchmap'],
            help="Model config file.")
    parser.add_argument("--restore-epoch", type=int, default=-1, help="Epoch to resore training from.")
    parser.add_argument("--image-base", default=".", help="Path to images.")
    command_line_args = parser.parse_args()

    # Setting up model specifics
    heading(f'\nSetting up model files ')
    args, image_base = modelSetup(command_line_args)

    world_size = torch.cuda.device_count()
    mp.spawn(
        spawn_training,
        args=(world_size, image_base, args),
        nprocs=world_size,
        join=True,
    )