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
from torch import nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from models.infonce import infonce
from models.multimodalModels import vision, mutlimodal
from models.setup import modelSetup
from models.util import heading
from training.util import (
    printEpoch,
    saveModelAttriburesAndTrainingAMP,
    timeFormat,
    valueTracking,
)

from mydata import LeanneDataset


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
        acc = (ind[:, 0:treshold] <= treshold).float().mean().detach().item()

        end_time = time.time()
        _, hours, minutes, seconds = timeFormat(start_time, end_time)
        print(f"Prediction accuracy: {acc*100}%")
        print(
            f"Validation took {hours:>2} hours {minutes:>2} minutes {seconds:>2} seconds"
        )

    return acc


class MattNet(nn.Module):
    def __init__(self, args):
        super(MattNet, self).__init__()
        self.audio_enc = mutlimodal(args)
        self.image_enc = vision(args)



def myloss(
    image_output,
    english_output,
    dutch_output,
    french_output,
    negatives,
    *args,
    **kwargs,
):
    from torch.nn import functional as F

    def compute1(i, a):
        att = torch.bmm(a.transpose(1, 2), i)
        s = att.max()
        return s.unsqueeze(0).unsqueeze(0)

    def loss_dir(pred):
        # pred1 = torch.concat((pred[0, :1], pred[0, true == 0]))
        pred1 = pred[0]
        pred1 = F.softmax(pred1, dim=0)
        loss = -pred1[0].log()
        return loss
        # pred1 = pred[true == 1]
        # pred1 = F.softmax(pred1, dim=1)
        # pred1 = pred1[:, true == 1].sum(dim=1)
        # loss = -torch.log(pred1).mean()
        # return loss

    pred_a_v = compute1(image_output, english_output)
    pred_aneg_v = [compute1(image_output, neg["english_output"]) for neg in negatives]
    pred_aneg_v = torch.cat(pred_aneg_v, dim=1)
    pred_a_vneg = [compute1(neg["image"], english_output) for neg in negatives]
    pred_a_vneg = torch.cat(pred_a_vneg, dim=1)

    pred1 = torch.cat((pred_a_v, pred_a_vneg), dim=1)
    loss1 = loss_dir(pred1)

    pred2 = torch.cat((pred_a_v, pred_aneg_v), dim=1)
    loss2 = loss_dir(pred2)

    return (loss1 + loss2) / 2


def train1(image_base, args):

    device = "cuda"
    torch.manual_seed(42)
    writer = SummaryWriter(args["exp_dir"] / "tensorboard")

    best_epoch, best_acc = 0, 0
    global_step, start_epoch = 0, 0
    info = {}
    rank = 0
    loss_tracker = valueTracking()

    heading("\nLoading training data ")
    train_dataset = LeanneDataset("train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )

    heading("\nLoading validation data ")

    args["image_config"]["center_crop"] = True
    valid_dataset = LeanneDataset("train")
    validation_loader = DataLoader(
        valid_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    from mymme.model import MattNet

    # model = MattNet(args).to(rank)
    model = MattNet({}, {}).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        args["learning_rate_scheduler"]["initial_learning_rate"],
        weight_decay=args["weight_decay"],
    )

    start_epoch += 1

    for epoch in np.arange(start_epoch, args["n_epochs"] + 1):
        # train_dataset.set_epoch(int(epoch))
        # current_learning_rate = adjust_learning_rate(args, optimizer, epoch, 0.00001)
        current_learning_rate = args["learning_rate_scheduler"]["initial_learning_rate"]

        model.train()
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

        def loss_dir(pred):
            from torch.nn import functional as F
            pred1 = pred[0]
            pred1 = F.softmax(pred1, dim=0)
            loss = -pred1[0].log()
            return loss

        for value_dict in train_loader:

            optimizer.zero_grad()

            image_inp = torch.cat(
                [value_dict["image"]] + 
                [datum["neg_image"] for datum in value_dict["negatives"]],
                dim=0
            )

            audio_inp = torch.cat(
                [value_dict["english_feat"]] + 
                [datum["neg_english"] for datum in value_dict["negatives"]],
                dim=0
            )

            # image_emb = model.image_enc(image_inp.to(rank))
            # audio_emb = model.audio_enc(audio_inp.to(rank))[-1]

            # op = "xda,ydi->xyai"
            # sim = torch.einsum(op, audio_emb, image_emb)
            # sim, _ = sim.max(dim=-1)
            # sim, _ = sim.max(dim=-1)

            sim = model(audio_inp.to(device), image_inp.to(device))

            loss1 = loss_dir(sim)
            loss2 = loss_dir(sim.T)
            loss = (loss1 + loss2) / 2

            loss.backward()
            optimizer.step()

            loss_val = loss.detach().item()
            loss_tracker.update(loss_val, 1)
            end_time = time.time()
            # print(loss.item())
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

        import pdb; pdb.set_trace()
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
