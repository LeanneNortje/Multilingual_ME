#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import argparse
import os
import pickle
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from image_caption_dataset_preprocessing import spokenData
import json
from pathlib import Path
import numpy
from collections import Counter
import sys
from os import popen
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial

terminal_rows, terminal_width = popen('stty size', 'r').read().split()
terminal_width = int(terminal_width)
def heading(string):
    print("_"*terminal_width + "\n")
    print("-"*10 + string + "-"*10 + "\n")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--image-base", default="..", help="Path to images.")
command_line_args = parser.parse_args()

image_base = Path(command_line_args.image_base).absolute()

with open("preprocessing_spokencoco_config.json") as file:
  args = json.load(file)

args["data_train"] = image_base / args["data_train"]
args["data_val"] = image_base / args["data_val"]

args["audio_base"] = image_base / args["audio_base"]
args["out_dir"] = Path(args["out_dir"])

if not os.path.isdir((Path("..") / args["out_dir"]).absolute()):
    (Path("..") / args["out_dir"]).absolute().mkdir(parents=True)


# Load in txt files
#_________________________________________________________________________________________________

def load(fn, langauge):

    # with open(json_fn, 'r') as f:
    #     raw_data = json.load(f)
    # heading(f'Loading in {json_fn.stem} ')

    # data = []
    # for audio_fn in tqdm(raw_data): 
    #     spkr = list(set(raw_data[audio_fn]['speaker']))[0]                           
    #     point = {
    #         "speaker": spkr,
    #         "wav": audio_fn,
    #         "ids": raw_data[audio_fn]['ids']
    #     }
    #     data.append(point)

    raw_data = np.load(fn, allow_pickle=True)['lookup'].item()
    data = []

    wav_to_id = {}
    for id in raw_data:
        for wav_name in raw_data[id][langauge]:
            for wav in raw_data[id][langauge][wav_name]:
                if wav not in wav_to_id: wav_to_id[wav] = []
                wav_to_id[wav].append(id)

    for wav in wav_to_id:
        print(wav)
        point = {
            "wav": wav,
            "ids": wav_to_id[wav]
        }
        data.append(point)
    

    print(f'{len(data)} data points in {fn}\n')
    return data

langauge = 'french'    
train_data = load(args['data_train'], langauge)
val_data = load(args['data_val'], langauge)

lengths = []

def saveADatapoint(ids, save_fn, audio_feat, args):
    ids = list(set(ids))
    ids = [int(id) for id in ids]
    numpy.savez_compressed(
        save_fn.absolute(), 
        ids=ids,
        audio_feat=audio_feat.squeeze().numpy()
        )
    # print(list(np.load(str(save_fn) + '.npz', allow_pickle=True)['ids']))
    return "/".join(str(save_fn).split("/")[1:])

def SaveDatapoints(dataloader, subset, datasets):

    ouputs = []
    executor = ProcessPoolExecutor(max_workers=cpu_count()) 
    
    save_fn = Path("..") / args["out_dir"] / Path(datasets)
    if not save_fn.absolute().is_dir(): 
        save_fn.absolute().mkdir(parents=True)
        print(f'Made {save_fn}.')

    for i, (audio_feat, audio_name, ids, frames) in enumerate(tqdm(dataloader, leave=False)):
        lengths.append(frames.item())
        this_save_fn = save_fn / str(audio_name[0])  
        ouputs.append(executor.submit(
            partial(saveADatapoint, ids[0], this_save_fn, audio_feat, args)))


    data_paths = [entry.result() for entry in tqdm(ouputs)]
    json_fn = (Path("..") / args["out_dir"]).absolute() / Path(datasets + "_" + subset + ".json")      
    with open(json_fn, "w") as json_file: json.dump(data_paths, json_file, indent="")
    print(f'Wrote {len(data_paths)} data points to {json_fn}.')

    return json_fn      

heading(f'Preprocessing training data points.')
train_loader = torch.utils.data.DataLoader(
    spokenData(train_data, args['audio_base'] / Path(f'{langauge}_words'), args["audio_config"], True),
    batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
train_json_fn = SaveDatapoints(train_loader, "train", langauge)

heading(f'Preprocessing validation data points.')
# args["image_config"]["center_crop"] = True
val_loader = torch.utils.data.DataLoader(
    spokenData(val_data, args['audio_base'] / Path(f'{langauge}_words'), args["audio_config"]),
    batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
val_json_fn = SaveDatapoints(val_loader, "val", langauge)

print(np.min(lengths))
print(np.mean(lengths))
print(np.max(lengths))