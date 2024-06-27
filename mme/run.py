# _________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
# _________________________________________________________________________________________________
# adapted from https://github.com/dharwath

import argparse
import time

from pathlib import Path

import numpy as np

from tqdm import tqdm
from PIL import Image

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


from dataloaders import ImageAudioDatawithSampling, ImageAudioDatawithSamplingVal
from models.GeneralModels import ScoringAttentionModule
from models.infonce import infonce
from models.multimodalModels import vision, mutlimodal
from models.setup import modelSetup
from models.util import heading, loadPretrainedWeights
from losses import compute_matchmap_similarity_matrix_loss
from training.util import (
    adjust_learning_rate,
    getParameters,
    loadModelAttriburesAndTrainingAMP,
    loadModelAttriburesAndTrainingAtEpochAMP,
    printEpoch,
    saveModelAttriburesAndTrainingAMP,
    timeFormat,
    valueTracking,
)


BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"


def validate(
    audio_model,
    image_model,
    attention,
    contrastive_loss,
    val_loader,
    rank,
    image_base,
    args,
):
    # function adapted from https://github.com/dharwath
    start_time = time.time()
    anch = []
    # positives = []
    negatives = []

    with torch.no_grad():

        for value_dict in tqdm(val_loader, leave=False):

            image_output = image_model(value_dict["image"].to(rank))

            english_input = value_dict["english_feat"].to(rank)
            _, _, english_output = audio_model(english_input)
            anch_s = attention.score(image_output, english_output)
            anch.append(anch_s)

            # ps = []
            # for p, pos_dict in enumerate(value_dict['positives']):
            #     pos_image_output = image_model(pos_dict["pos_image"].to(rank))
            #     score = attention.score(pos_image_output, english_output)
            #     ps.append(score)
            # positives.append(torch.cat(ps, dim=1))

            ns = []
            for neg_dict in value_dict["negatives"]:
                neg_image_output = image_model(neg_dict["neg_image"].to(rank))
                score = attention.score(neg_image_output, english_output)
                ns.append(score)
            negatives.append(torch.cat(ns, dim=1))

        anch = torch.cat(anch, dim=0)
        # positives = torch.cat(positives, dim=0)
        negatives = torch.cat(negatives, dim=0)
        scores = torch.cat([anch, negatives], dim=1)
        treshold = anch.size(1)  # + positives.size(1)
        ind = torch.argsort(scores, dim=1, descending=True)[:, 0:treshold]
        acc = (ind[:, 0:treshold] < treshold).float().mean().detach().item()

        end_time = time.time()
        _, hours, minutes, seconds = timeFormat(start_time, end_time)
        print(f"Prediction accuracy: {acc*100}%")
        print(
            f"Validation took {hours:>2} hours {minutes:>2} minutes {seconds:>2} seconds"
        )

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

    if rank == 0:
        writer = SummaryWriter(args["exp_dir"] / "tensorboard")

    best_epoch, best_acc = 0, 0
    global_step, start_epoch = 0, 0
    info = {}
    loss_tracker = valueTracking()

    if rank == 0:
        heading("\nLoading training data ")
    train_dataset = ImageAudioDatawithSampling(
        image_base, args["data_train"], Path("data/train_lookup.npz"), args, rank
    )
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
        heading("\nLoading validation data ")
        args["image_config"]["center_crop"] = True
        validation_loader = torch.utils.data.DataLoader(
            ImageAudioDatawithSamplingVal(
                image_base, args["data_val"], Path("data/val_lookup.npz"), args, rank
            ),
            batch_size=args["batch_size"],
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

    if rank == 0:
        heading("\nSetting up Audio model ")
    audio_model = mutlimodal(args).to(rank)

    if rank == 0:
        heading("\nSetting up image model ")
    image_model = vision(args).to(rank)

    if rank == 0:
        heading("\nSetting up attention model ")
    attention = ScoringAttentionModule(args).to(rank)

    if rank == 0:
        heading("\nSetting up contrastive loss ")
    contrastive_loss = infonce

    model_with_params_to_update = {
        "audio_model": audio_model,
        "attention": attention,
        # "contrastive_loss": contrastive_loss,
        "image_model": image_model,
    }
    model_to_freeze = {}
    trainable_parameters = getParameters(
        model_with_params_to_update, model_to_freeze, args
    )

    if args["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            trainable_parameters,
            args["learning_rate_scheduler"]["initial_learning_rate"],
            momentum=args["momentum"],
            weight_decay=args["weight_decay"],
        )
    elif args["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            trainable_parameters,
            args["learning_rate_scheduler"]["initial_learning_rate"],
            weight_decay=args["weight_decay"],
        )
    else:
        raise ValueError("Optimizer %s is not supported" % args["optimizer"])

    audio_model = DDP(audio_model, device_ids=[rank])
    image_model = DDP(image_model, device_ids=[rank])

    if args["resume"] is False and args["cpc"]["warm_start"]:
        if rank == 0:
            print("Loading pretrained acoustic weights")
        audio_model = loadPretrainedWeights(audio_model, args, rank)

    if args["resume"]:

        if "restore_epoch" in args:
            info, start_epoch, global_step, best_epoch, best_acc = (
                loadModelAttriburesAndTrainingAtEpochAMP(
                    args["exp_dir"],
                    audio_model,
                    image_model,
                    attention,
                    contrastive_loss,
                    optimizer,
                    rank,
                    args["restore_epoch"],
                )
            )
            if rank == 0:
                print(
                    f"\nEpoch particulars:\n\t\tepoch = {start_epoch}\n\t\tglobal_step = {global_step}\n\t\tbest_epoch = {best_epoch}\n\t\tbest_acc = {best_acc}\n"
                )
        else:
            info, start_epoch, global_step, best_epoch, best_acc = (
                loadModelAttriburesAndTrainingAMP(
                    args["exp_dir"],
                    audio_model,
                    image_model,
                    attention,
                    contrastive_loss,
                    optimizer,
                    rank,
                )
            )
            if rank == 0:
                print(
                    f"\nEpoch particulars:\n\t\tepoch = {start_epoch}\n\t\tglobal_step = {global_step}\n\t\tbest_epoch = {best_epoch}\n\t\tbest_acc = {best_acc}\n"
                )

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
        if rank == 0:
            printEpoch(
                epoch,
                0,
                len(train_loader),
                loss_tracker,
                best_acc,
                start_time,
                start_time,
                current_learning_rate,
            )

        i = 0

        for value_dict in train_loader:

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():

                image_output = image_model(value_dict["image"].to(rank))

                english_input = value_dict["english_feat"].to(rank)
                _, _, english_output = audio_model(english_input)

                dutch_input = value_dict["dutch_feat"].to(rank)
                _, _, dutch_output = audio_model(dutch_input)

                french_input = value_dict["french_feat"].to(rank)
                _, _, french_output = audio_model(french_input)

                positives = []
                for p, pos_dict in enumerate(value_dict["positives"]):
                    pos_image_output = image_model(
                        pos_dict["pos_image"].to(rank)
                    )  # pos_images[p]

                    pos_english_input = pos_dict["pos_english"].to(rank)
                    _, _, pos_english_output = audio_model(pos_english_input)

                    pos_dutch_input = pos_dict["pos_dutch"].to(rank)
                    _, _, pos_dutch_output = audio_model(pos_dutch_input)

                    pos_french_input = pos_dict["pos_french"].to(rank)
                    _, _, pos_french_output = audio_model(pos_french_input)

                    positives.append(
                        {
                            "image": pos_image_output,
                            "english_output": pos_english_output,
                            "dutch_output": pos_dutch_output,
                            "french_output": pos_french_output,
                        }
                    )

                negatives = []
                for n, neg_dict in enumerate(value_dict["negatives"]):
                    neg_image_output = image_model(
                        neg_dict["neg_image"].to(rank)
                    )  # neg_images[n]

                    neg_english_input = neg_dict["neg_english"].to(rank)
                    _, _, neg_english_output = audio_model(neg_english_input)

                    neg_dutch_input = neg_dict["neg_dutch"].to(rank)
                    _, _, neg_dutch_output = audio_model(neg_dutch_input)

                    neg_french_input = neg_dict["neg_french"].to(rank)
                    _, _, neg_french_output = audio_model(neg_french_input)

                    negatives.append(
                        {
                            "image": neg_image_output,
                            "english_output": neg_english_output,
                            "dutch_output": neg_dutch_output,
                            "french_output": neg_french_output,
                        }
                    )

                loss = compute_matchmap_similarity_matrix_loss(
                    image_output,
                    english_output,
                    dutch_output,
                    french_output,
                    negatives,
                    positives,
                    attention,
                    contrastive_loss,  # audio_attention,
                    margin=args["margin"],
                    simtype=args["simtype"],
                    alphas=args["alphas"],
                    rank=rank,
                )

            loss.backward()
            optimizer.step()

            loss_tracker.update(
                loss.detach().item(), english_input.detach().size(0)
            )  #####
            end_time = time.time()
            if rank == 0:
                printEpoch(
                    epoch,
                    i + 1,
                    len(train_loader),
                    loss_tracker,
                    best_acc,
                    start_time,
                    end_time,
                    current_learning_rate,
                )
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
            avg_acc = validate(
                audio_model,
                image_model,
                attention,
                contrastive_loss,
                validation_loader,
                rank,
                image_base,
                args,
            )
            # break
            writer.add_scalar("loss/train", loss_tracker.average, epoch)
            writer.add_scalar("loss/val", avg_acc, epoch)

            best_acc, best_epoch = saveModelAttriburesAndTrainingAMP(
                args["exp_dir"],
                audio_model,
                image_model,
                attention,
                contrastive_loss,
                optimizer,
                info,
                int(epoch),
                global_step,
                best_epoch,
                avg_acc,
                best_acc,
                loss_tracker.average,
                end_time - start_time,
            )
        # break
    dist.destroy_process_group()


def train1(image_base, args):

    torch.manual_seed(42)
    writer = SummaryWriter(args["exp_dir"] / "tensorboard")

    best_epoch, best_acc = 0, 0
    global_step, start_epoch = 0, 0
    info = {}
    rank = 0
    loss_tracker = valueTracking()

    heading("\nLoading training data ")
    train_dataset = ImageAudioDatawithSampling(
        image_base,
        args["data_train"],
        Path("data/train_lookup.npz"),
        args,
        rank,
    )
    # subset
    num_samples = len(train_dataset)
    # train_dataset = torch.utils.data.Subset(train_dataset, range(0, num_samples, 100))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    heading("\nLoading validation data ")
    args["image_config"]["center_crop"] = True
    validation_loader = torch.utils.data.DataLoader(
        ImageAudioDatawithSamplingVal(
            image_base,
            args["data_val"],
            Path("data/val_lookup.npz"),
            args,
            rank,
        ),
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    heading("\nSetting up Audio model ")
    audio_model = mutlimodal(args).to(rank)

    heading("\nSetting up image model ")
    image_model = vision(args).to(rank)

    heading("\nSetting up attention model ")
    attention = ScoringAttentionModule(args).to(rank)

    heading("\nSetting up contrastive loss ")
    contrastive_loss = infonce

    model_with_params_to_update = {
        "audio_model": audio_model,
        "attention": attention,
        # "contrastive_loss": contrastive_loss,
        "image_model": image_model,
    }
    model_to_freeze = {}
    trainable_parameters = getParameters(
        model_with_params_to_update, model_to_freeze, args
    )

    if args["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            trainable_parameters,
            args["learning_rate_scheduler"]["initial_learning_rate"],
            momentum=args["momentum"],
            weight_decay=args["weight_decay"],
        )
    elif args["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            trainable_parameters,
            args["learning_rate_scheduler"]["initial_learning_rate"],
            weight_decay=args["weight_decay"],
        )
    else:
        raise ValueError("Optimizer %s is not supported" % args["optimizer"])

    if args["resume"] is False and args["cpc"]["warm_start"]:
        print("Loading pretrained acoustic weights")
        audio_model = loadPretrainedWeights(audio_model, args, rank)

    if args["resume"]:
        if "restore_epoch" in args:
            info, start_epoch, global_step, best_epoch, best_acc = (
                loadModelAttriburesAndTrainingAtEpochAMP(
                    args["exp_dir"],
                    audio_model,
                    image_model,
                    attention,
                    contrastive_loss,
                    optimizer,
                    rank,
                    args["restore_epoch"],
                )
            )
            print(
                f"\nEpoch particulars:\n\t\tepoch = {start_epoch}\n\t\tglobal_step = {global_step}\n\t\tbest_epoch = {best_epoch}\n\t\tbest_acc = {best_acc}\n"
            )
        else:
            info, start_epoch, global_step, best_epoch, best_acc = (
                loadModelAttriburesAndTrainingAMP(
                    args["exp_dir"],
                    audio_model,
                    image_model,
                    attention,
                    contrastive_loss,
                    optimizer,
                    rank,
                )
            )
            print(
                f"\nEpoch particulars:\n\t\tepoch = {start_epoch}\n\t\tglobal_step = {global_step}\n\t\tbest_epoch = {best_epoch}\n\t\tbest_acc = {best_acc}\n"
            )

    start_epoch += 1

    avg_acc = validate(
        audio_model,
        image_model,
        attention,
        contrastive_loss,
        validation_loader,
        rank,
        image_base,
        args,
    )

    for epoch in np.arange(start_epoch, args["n_epochs"] + 1):
        # train_dataset.set_epoch(int(epoch))
        current_learning_rate = adjust_learning_rate(args, optimizer, epoch, 0.00001)

        audio_model.train()
        image_model.train()
        attention.train()
        # contrastive_loss.train()

        loss_tracker.new_epoch()
        start_time = time.time()
        printEpoch(
            epoch,
            0,
            len(train_loader),
            loss_tracker,
            best_acc,
            start_time,
            start_time,
            current_learning_rate,
        )

        i = 0

        for value_dict in train_loader:

            optimizer.zero_grad()

            image_output = image_model(value_dict["image"].to(rank))

            english_input = value_dict["english_feat"].to(rank)
            _, _, english_output = audio_model(english_input)

            dutch_input = value_dict["dutch_feat"].to(rank)
            _, _, dutch_output = audio_model(dutch_input)

            french_input = value_dict["french_feat"].to(rank)
            _, _, french_output = audio_model(french_input)

            positives = []
            for p, pos_dict in enumerate(value_dict["positives"]):
                pos_image_output = image_model(pos_dict["pos_image"].to(rank)) 

                pos_english_input = pos_dict["pos_english"].to(rank)
                _, _, pos_english_output = audio_model(pos_english_input)

                pos_dutch_input = pos_dict["pos_dutch"].to(rank)
                _, _, pos_dutch_output = audio_model(pos_dutch_input)

                pos_french_input = pos_dict["pos_french"].to(rank)
                _, _, pos_french_output = audio_model(pos_french_input)

                positives.append(
                    {
                        "image": pos_image_output,
                        "english_output": pos_english_output,
                        "dutch_output": pos_dutch_output,
                        "french_output": pos_french_output,
                    }
                )

            negatives = []
            for n, neg_dict in enumerate(value_dict["negatives"]):
                neg_image_output = image_model(
                    neg_dict["neg_image"].to(rank)
                )  # neg_images[n]

                neg_english_input = neg_dict["neg_english"].to(rank)
                _, _, neg_english_output = audio_model(neg_english_input)

                neg_dutch_input = neg_dict["neg_dutch"].to(rank)
                _, _, neg_dutch_output = audio_model(neg_dutch_input)

                neg_french_input = neg_dict["neg_french"].to(rank)
                _, _, neg_french_output = audio_model(neg_french_input)

                negatives.append(
                    {
                        "image": neg_image_output,
                        "english_output": neg_english_output,
                        "dutch_output": neg_dutch_output,
                        "french_output": neg_french_output,
                    }
                )

            loss = compute_matchmap_similarity_matrix_loss(
                image_output,
                english_output,
                dutch_output,
                french_output,
                negatives,
                positives,
                attention,
                contrastive_loss,  # audio_attention,
                margin=args["margin"],
                simtype=args["simtype"],
                alphas=args["alphas"],
                rank=rank,
            )

            loss.backward()
            optimizer.step()

            loss_tracker.update(
                loss.detach().item(), english_input.detach().size(0)
            )  #####
            end_time = time.time()
            printEpoch(
                epoch,
                i + 1,
                len(train_loader),
                loss_tracker,
                best_acc,
                start_time,
                end_time,
                current_learning_rate,
            )
            if np.isnan(loss_tracker.average):
                print("training diverged...")
                return

            global_step += 1
            i += 1

        avg_acc = validate(
            audio_model,
            image_model,
            attention,
            contrastive_loss,
            validation_loader,
            rank,
            image_base,
            args,
        )

        writer.add_scalar("loss/train", loss_tracker.average, epoch)
        writer.add_scalar("loss/val", avg_acc, epoch)

        best_acc, best_epoch = saveModelAttriburesAndTrainingAMP(
            args["exp_dir"],
            audio_model,
            image_model,
            attention,
            contrastive_loss,
            optimizer,
            info,
            int(epoch),
            global_step,
            best_epoch,
            avg_acc,
            best_acc,
            loss_tracker.average,
            end_time - start_time,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        dest="resume",
        help="load from exp_dir if True",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="matchmap",
        choices=["matchmap"],
        help="Model config file.",
    )
    parser.add_argument(
        "--restore-epoch",
        type=int,
        default=-1,
        help="Epoch to resore training from.",
    )
    parser.add_argument(
        "--image-base",
        default=".",
        help="Path to images.",
    )
    command_line_args = parser.parse_args()

    # Setting up model specifics
    heading("\nSetting up model files ")
    args, image_base = modelSetup(command_line_args)

    train1(image_base, args)
    # world_size = torch.cuda.device_count() - 2
    # mp.spawn(
    #     spawn_training,
    #     args=(world_size, image_base, args),
    #     nprocs=world_size,
    #     join=True,
    # )
