import argparse
import pickle

from pathlib import Path

import librosa
import numpy as np
import scipy
import scipy.signal

from PIL import Image
from tqdm import tqdm

import torch
from torch.nn.parallel import DataParallel as DDP
from torchvision import transforms

from models.util import heading
from models.GeneralModels import ScoringAttentionModule
from models.infonce import infonce
from models.multimodalModels import mutlimodal, vision
from training.util import getParameters, loadModelAttriburesAndTrainingAMP, loadModelAttriburesAndTrainingAtEpochAMP



config_library = {
    "multilingual": "English_Hindi_DAVEnet_config.json",
    "multilingual+matchmap": "English_Hindi_matchmap_DAVEnet_config.json",
    "english": "English_DAVEnet_config.json",
    "english+matchmap": "English_matchmap_DAVEnet_config.json",
    "hindi": "Hindi_DAVEnet_config.json",
    "hindi+matchmap": "Hindi_matchmap_DAVEnet_config.json",
}

scipy_windows = {
    "hamming": scipy.signal.hamming,
    "hann": scipy.signal.hann,
    "blackman": scipy.signal.blackman,
    "bartlett": scipy.signal.bartlett,
}


def myRandomCrop(im, resize, to_tensor):
    im = resize(im)
    im = to_tensor(im)
    return im


def preemphasis(signal, coeff=0.97):
    # function adapted from https://github.com/dharwath
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def read_file(path, parse_fn=lambda x: x.strip()):
    with open(path, "r") as f:
        return [parse_fn(line) for line in f.readlines()]


episodes = np.load(Path("data/episodes.npz"), allow_pickle=True)["episodes"].item()
fam_1 = []
fam_2 = []
me = []
fam_me = []

print("\n\n")
rewind = "\033[A" * 2
c = 0
t = 0
with torch.no_grad():

    audio_datapoints = np.load(
        Path("results/files/episode_data.npz"), allow_pickle=True
    )["audio_datapoints"].item()
    image_datapoints = np.load(
        Path("results/files/episode_data.npz"), allow_pickle=True
    )["image_datapoints"].item()

    query = np.load(Path("results/files/episode_data.npz"), allow_pickle=True)[
        "audio_1"
    ].item()
    query_labels = np.load(Path("results/files/episode_data.npz"), allow_pickle=True)[
        "audio_labels_1"
    ].item()
    query_tag = np.load(Path("results/files/episode_data.npz"), allow_pickle=True)[
        "audio_tag_1"
    ].item()

    secondary_query = np.load(
        Path("results/files/episode_data.npz"), allow_pickle=True
    )["audio_2"].item()
    secondary_labels = np.load(
        Path("results/files/episode_data.npz"), allow_pickle=True
    )["audio_labels_2"].item()
    secondary_tag = np.load(Path("results/files/episode_data.npz"), allow_pickle=True)[
        "audio_tag_2"
    ].item()

    image_1 = np.load(Path("results/files/episode_data.npz"), allow_pickle=True)[
        "image_1"
    ].item()
    image_2 = np.load(Path("results/files/episode_data.npz"), allow_pickle=True)[
        "image_2"
    ].item()
    image_labels_1 = np.load(Path("results/files/episode_data.npz"), allow_pickle=True)[
        "image_labels_1"
    ].item()
    image_labels_2 = np.load(Path("results/files/episode_data.npz"), allow_pickle=True)[
        "image_labels_2"
    ].item()
    image_tag_1 = np.load(Path("results/files/episode_data.npz"), allow_pickle=True)[
        "image_tag_1"
    ].item()
    image_tag_2 = np.load(Path("results/files/episode_data.npz"), allow_pickle=True)[
        "image_tag_2"
    ].item()

    words_seen = read_file("data/seen.txt")
    words_unseen = read_file("data/unseen.txt")

    def map_labels_to_types(labels):
        def map1(label):
            if label in words_seen:
                return "familiar"
            elif label in words_unseen:
                return "novel"
            else:
                assert False, f"Unknown label {label}"
        return [map1(label) for label in labels]

    import pandas as pd

    # import pdb; pdb.set_trace()

    # episodes = np.load(Path('data/episodes.npz'), allow_pickle=True)['episodes'].item()
    # model_metadata_dir = Path("/home/doneata/data/multilingual-mutual-exclusivity-bias/checkpoints/english/model_metadata")
    model_metadata_base = Path("/home/doneata/work/multilingual-me/trilingual_no_language_links/")
    for args_fn in (model_metadata_base / "model_metadata" / "0a0057c11d").rglob("*/args.pkl"):
        one_familiar_results = {"correct": 0, "total": 0}
        two_familiar_results = {"correct": 0, "total": 0}
        novel_results = {"correct": 0, "total": 0}
        known_novel_results = {"correct": 0, "total": 0}
        per_novel_word = {}
        per_familiar_word = {}
        per_novel_word_faults = {}

        with open(args_fn, "rb") as f:
            args = pickle.load(f)

            # Setting up model specifics
        # heading(f'\nSetting up model files ')
        # args, image_base = modelSetup(command_line_args, True)
        rank = "cuda"

        audio_conf = args["audio_config"]
        target_length = audio_conf.get("target_length", 128)
        padval = audio_conf.get("padval", 0)
        image_conf = args["image_config"]
        crop_size = image_conf.get("crop_size")
        center_crop = image_conf.get("center_crop")
        RGB_mean = image_conf.get("RGB_mean")
        RGB_std = image_conf.get("RGB_std")

        # image_resize_and_crop = transforms.Compose(
        #         [transforms.Resize(224), transforms.ToTensor()])
        resize = transforms.Resize((256, 256))
        to_tensor = transforms.ToTensor()
        image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)
        image_resize = transforms.transforms.Resize((256, 256))
        trans = transforms.ToPILImage()

        # Create models
        audio_model = mutlimodal(args).to(rank)
        image_model = vision(args).to(rank)
        attention = ScoringAttentionModule(args).to(rank)
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
        else:
            heading("\nRetoring model parameters from best epoch ")
            info, epoch, global_step, best_epoch, best_acc = (
                loadModelAttriburesAndTrainingAMP(
                    model_metadata_base / args["exp_dir"],
                    audio_model,
                    image_model,
                    attention,
                    contrastive_loss,
                    optimizer,
                    rank,
                    False,
                )
            )

        for ep_num in tqdm(episodes, desc=f"{rewind}"):
            # if ep_num != 25: continue
            episode = episodes[ep_num]

            query_output = []
            for name in query[ep_num]:
                # print(name)
                query_output.append(audio_datapoints[name])
            query_output = torch.cat(query_output, dim=0)
            _, _, query_output = audio_model(query_output.to(rank))
            # break

            query_image = []
            for name in image_1[ep_num]:
                query_image.append(image_datapoints[name])
            query_image = torch.cat(query_image, dim=0)
            query_image = image_model(query_image.to(rank))

            secondary_query_output = []
            for name in secondary_query[ep_num]:
                secondary_query_output.append(audio_datapoints[name])
            secondary_query_output = torch.cat(secondary_query_output, dim=0)
            _, _, secondary_query_output = audio_model(secondary_query_output.to(rank))

            other_image = []
            for name in image_2[ep_num]:
                other_image.append(image_datapoints[name])
            other_image = torch.cat(other_image, dim=0)
            other_image = image_model(other_image.to(rank))

            tags_1 = query_tag[ep_num]
            labels_1 = query_labels[ep_num]
            tags_2 = secondary_tag[ep_num]
            labels_2 = secondary_labels[ep_num]

            df = {
                "audio-1": map_labels_to_types(labels_1),
                "audio-2": map_labels_to_types(labels_2),
                "image-1": map_labels_to_types(image_labels_1[ep_num]),
                "image-2": map_labels_to_types(image_labels_2[ep_num]),
                "audio-labels-1": labels_1,
                "audio-labels-2": labels_2,
                "image-labels-1": image_labels_1[ep_num],
                "image-labels-2": image_labels_2[ep_num],
            }
            df = pd.DataFrame(df)
            print(df)

            import pdb; pdb.set_trace()

            for i in range(query_output.size(0)):

                images = torch.cat(
                    [
                        query_image[i, :, :].unsqueeze(0),
                        other_image[i, :, :].unsqueeze(0),
                    ],
                    dim=0,
                )

                scores = attention.one_to_many_score(
                    images, query_output[i, :, :].unsqueeze(0)
                ).squeeze()
                index = torch.argmax(scores).item()

                if tags_1[i] == "familiar_1":
                    if labels_1[i] not in per_familiar_word:
                        per_familiar_word[labels_1[i]] = {"correct": 0, "total": 0}
                    if index == 0:
                        one_familiar_results["correct"] += 1
                        per_familiar_word[labels_1[i]]["correct"] += 1
                    one_familiar_results["total"] += 1
                    per_familiar_word[labels_1[i]]["total"] += 1

                elif tags_1[i] == "novel":
                    if labels_1[i] not in per_novel_word:
                        per_novel_word[labels_1[i]] = {"correct": 0, "total": 0}
                    if labels_1[i] not in per_novel_word_faults:
                        per_novel_word_faults[labels_1[i]] = {}
                    if index == 0:
                        novel_results["correct"] += 1
                        per_novel_word[labels_1[i]]["correct"] += 1
                        c += 1
                    else:
                        if labels_2[i] not in per_novel_word_faults[labels_1[i]]:
                            per_novel_word_faults[labels_1[i]][labels_2[i]] = 0
                        per_novel_word_faults[labels_1[i]][labels_2[i]] += 1
                    novel_results["total"] += 1
                    per_novel_word[labels_1[i]]["total"] += 1
                    t += 1

                scores = attention.one_to_many_score(
                    images, secondary_query_output[i, :, :].unsqueeze(0)
                ).squeeze()
                index = torch.argmax(scores).item()

                if tags_2[i] == "familiar_2":
                    if labels_2[i] not in per_familiar_word:
                        per_familiar_word[labels_2[i]] = {"correct": 0, "total": 0}
                    if index == 1:
                        two_familiar_results["correct"] += 1
                        per_familiar_word[labels_2[i]]["correct"] += 1
                    two_familiar_results["total"] += 1
                    per_familiar_word[labels_2[i]]["total"] += 1
                elif tags_2[i] == "known_novel":
                    if index == 1:
                        known_novel_results["correct"] += 1
                    known_novel_results["total"] += 1
            print(f"{c}/{t}={100*c/t}%")
            # break

        fam_1.append(one_familiar_results)
        fam_2.append(two_familiar_results)
        me.append(novel_results)
        fam_me.append(known_novel_results)


results = {"familiar_1": fam_1, "familiar_2": fam_2, "novel": me, "known_novel": fam_me}
for r in results:
    scores = []
    for d in results[r]:
        # d = results[r]
        c = d["correct"]
        t = d["total"]
        p = round(100 * c / t, 2)
        print(f"{r:<12}: {c}/{t}={p:.2f}%")
        scores.append(p)

    m = np.mean(scores)
    v = np.std(scores)
    print(f"mean: {m:.2f}% std: {v:.2f}%\n")


# name = 'keyword'
# if args['cpc']['warm_start']: name += '_cpc'
# if args['pretrained_alexnet']: name += '_alexnet'
# name += f'_{args["instance"]}'

# f = open(Path(f'results/files/{name}.txt'), 'w')

# for w in per_novel_word:

#     d = per_novel_word[w]
#     c = d['correct']
#     t = d['total']
#     p = round(100*c/t, 2)
#     print(f'{w:<12}: {c}/{t}={p:.2f}%')
#     f.write(f'{w:<12} {p}\n')

# print(f'Novel faults')
# for w in per_novel_word_faults:
#     d = per_novel_word_faults[w]
#     t = per_novel_word[w]['total']
#     for con in d:
#         p = 100*d[con]/t
#         print(f'{w:<12} -> {con:<12}: {d[con]}/{t}={p:.2f}')

# print(f'\nFamiliar accuarcies')
# for w in per_familiar_word:
#     d = per_familiar_word[w]
#     c = d['correct']
#     t = d['total']
#     p = 100*c/t
#     print(f'{w:<12}: {c}/{t}={p:.2f}%')
