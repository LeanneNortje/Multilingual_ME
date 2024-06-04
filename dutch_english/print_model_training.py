#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
# adapted from https://github.com/dharwath

import argparse
import torch
from dataloaders import *
from training import train
from models.setup import *
from models.util import *

# Commandline arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pkl", type=str, help="Model pl file.")
command_line_args = parser.parse_args()
print(command_line_args.pkl)
with open(command_line_args.pkl, "r") as file:
	data = json.load(file)

for i in data:
	print(f'Epoch: {i:<3} Global step: {data[i]["global_step"]:<6} Loss: {data[i]["loss"]:1.6f} Acc: {data[i]["acc"]:1.6f} Best epoch: {data[i]["best_epoch"]:<3} Best acc: {data[i]["best_acc"]:1.6f}')
	# break