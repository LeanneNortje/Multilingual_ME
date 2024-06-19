from itertools import groupby
from typing import Literal, Tuple

import random
import json

import librosa
import numpy as np
import scipy

import torch

from torch.utils.data import DataLoader, Dataset, IterableDataset, default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence

from PIL import Image
from torchvision import transforms

from toolz import concat, dissoc
from mymme.utils import read_file, read_json


Split = Literal["train", "valid", "test"]
Language = Literal["english", "dutch", "french"]


def load_dictionary():
    def parse_line(line):
        en, nl, fr = line.split()
        return {
            "english": en,
            "dutch": nl,
            "french": fr,
        }

    return read_file("mymme/data/concepts.txt", parse_line)


transform_image = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_image(datum: dict) -> torch.Tensor:
    name = datum["name"]
    path = f"data/images/{name}.jpg"
    img = Image.open(path).convert("RGB")
    return transform_image(img)


def load_audio(datum: dict) -> torch.Tensor:
    def preemphasis(signal, coeff=0.97):
        return np.append(signal[0], signal[1:] - coeff * signal[:-1])

    CONFIG = {
        "preemph-coef": 0.97,
        "sample-rate": 16_000,
        "window-size": 0.025,
        "window-stride": 0.01,
        "window": scipy.signal.hamming,
        "num-mel-bins": 40,
        "target-length": 256,
        "use-raw_length": False,
        "padval": 0,
        "fmin": 20,
    }

    name = datum["name"]
    lang = datum["lang"]
    path = f"data/{lang}_words/{name}.wav"
    y, sr = librosa.load(path, sr=CONFIG["sample-rate"])
    y = y - y.mean()
    y = preemphasis(y, CONFIG["preemph-coef"])

    n_fft = int(CONFIG["sample-rate"] * CONFIG["window-size"])
    win_length = n_fft
    hop_length = int(CONFIG["sample-rate"] * CONFIG["window-stride"])

    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=CONFIG["num-mel-bins"],
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hamming",
        fmin=CONFIG["fmin"],
    )

    logspec = librosa.power_to_db(melspec, ref=np.max)
    logspec = torch.tensor(logspec)
    logspec = logspec.T

    return logspec


def group_by_word(data):
    get_word = lambda datum: datum["word-en"]
    data1 = sorted(data, key=get_word)
    return {key: list(group) for key, group in groupby(data1, get_word)}


class MEDataset:
    def __init__(self, split: Split, langs: Tuple[Language]):
        self.split = split
        self.langs = langs

        self.words_seen = read_file("mymme/data/words-seen.txt")
        self.words_unseen = read_file("mymme/data/words-unseen.txt")

        image_files = read_json(f"mymme/data/filelists/image-{split}.json")
        audio_files = read_json(f"mymme/data/filelists/audio-{split}.json")
        audio_files = [datum for datum in audio_files if datum["lang"] in langs]

        self.word_to_images = group_by_word(image_files)
        self.word_to_audios = group_by_word(audio_files)


class PairedMEDataset(IterableDataset):
    def __init__(self, split, langs, n_pos: int, n_neg: int):
        super(PairedMEDataset).__init__()

        assert split in ("train", "valid")
        self.dataset = MEDataset(split, langs)

        self.n_pos = n_pos
        self.n_neg = n_neg

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        # print(worker_info.id)

        def sample_neg(data, word):
            words = set(self.dataset.words_seen) - set([word])
            words = random.choices(list(words), k=self.n_neg)
            return [random.choice(data[word]) for word in words]

        while True:
            word = random.choice(self.dataset.words_seen)

            images_pos = random.sample(self.dataset.word_to_images[word], self.n_pos)
            audios_pos = random.sample(self.dataset.word_to_audios[word], self.n_pos)

            for image_name, audio_name in zip(images_pos, audios_pos):
                yield {
                    "image": load_image(image_name),
                    "audio": load_audio(audio_name),
                    "label": 1,
                }

            images_neg = sample_neg(self.dataset.word_to_images, word)
            audios_neg = sample_neg(self.dataset.word_to_audios, word)

            for image_name, audio_name in zip(images_neg, audios_neg):
                yield {
                    "image": load_image(image_name),
                    "audio": load_audio(audio_name),
                    "label": 0,
                }


def setup_data(config):
    num_pos = config["num-pos"]
    num_neg = config["num-neg"]
    batch_size = num_pos + num_neg

    kwargs = {
        "langs": ("english",),
        "n_pos": num_pos,
        "n_neg": num_neg,
    }
    train_dataset = PairedMEDataset(split="train", **kwargs)
    valid_dataset = PairedMEDataset(split="valid", **kwargs)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config["num_workers"],
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=config["num_workers"],
    )

    return train_dataloader, valid_dataloader


def my_collate_fn(batch):
    audios = pad_sequence([datum["audio"] for datum in batch], batch_first=True)
    rest = [dissoc(datum, "audio") for datum in batch]
    rest = default_collate(rest)
    return {"audio": audios, **rest}


if __name__ == "__main__":
    n_pos = 4
    n_neg = 12
    batch_size = n_pos + n_neg
    dataset = PairedMEDataset(split="train", langs=("english",), n_pos=n_pos, n_neg=n_neg)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=3, collate_fn=my_collate_fn)
    for batch in dataloader:
        print(batch)
        import pdb

        pdb.set_trace()
