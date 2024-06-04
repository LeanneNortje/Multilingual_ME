#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2023
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________


import json
import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--instance", type=int, default=1)
    a = parser.parse_args()

    with open(f'orig_params.json', "r") as file:
        args = json.load(file)

    print(args["instance"])
    args["instance"] = a.instance
    print(args["instance"])

    with open(f'params.json', "w") as file:
        file.write(json.dumps(args, indent=4))