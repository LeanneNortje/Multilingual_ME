from itertools import groupby
from typing import Literal, Tuple

import json
import random
import pdb

import librosa
import numpy as np
import scipy

import torch

from torch.utils.data import DataLoader, Dataset, IterableDataset, default_collate
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


def get_image_path(datum: dict):
    name = datum["name"]
    return f"data/images/{name}.jpg"


def get_audio_path(datum: dict):
    name = datum["name"]
    lang = datum["lang"]
    return f"data/{lang}_words/{name}.wav"


def load_image(datum: dict) -> torch.Tensor:
    path = get_image_path(datum)
    img = Image.open(path).convert("RGB")
    return transform_image(img)


def load_audio(datum: dict) -> torch.Tensor:
    def preemphasis(signal, coeff=0.97):
        return np.append(signal[0], signal[1:] - coeff * signal[:-1])

    def pad_to_length(data, len_target, pad_value=0):
        # data: T × D
        len_pad = len_target - len(data)
        if len_pad > 0:
            return np.pad(
                data,
                ((0, len_pad), (0, 0)),
                "constant",
                constant_values=pad_value,
            )
        else:
            return data[:len_target]

    CONFIG = {
        "preemph-coef": 0.97,
        "sample-rate": 16_000,
        "window-size": 0.025,
        "window-stride": 0.01,
        "window": scipy.signal.hamming,
        "num-mel-bins": 40,
        "target-length": 256,
        "use-raw_length": False,
        "pad-value": 0,
        "fmin": 20,
    }

    path = get_audio_path(datum)
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

    # logspec = melspec
    logspec = librosa.power_to_db(melspec, ref=np.max)  # D × T

    logspec = logspec.T  # T × D
    logspec = pad_to_length(logspec, CONFIG["target-length"], CONFIG["pad-value"])
    logspec = logspec.T

    logspec = torch.tensor(logspec)
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


class PairedMEDataset(Dataset):
    def __init__(
        self,
        split,
        langs,
        num_pos: int,
        num_neg: int,
        # num_word_repeats: int,
        to_shuffle: bool = False,
    ):
        super(PairedMEDataset).__init__()

        assert split in ("train", "valid")
        self.dataset = MEDataset(split, langs)

        self.n_pos = num_pos
        self.n_neg = num_neg

        # num_word_repeats = num_word_repeats if split == "train" else 1
        # words_seen = self.dataset.words_seen
        # self.words = [word for word in words_seen for _ in range(num_word_repeats)]

        # Use Leanne's order
        self.word_audio = [
            (word, audio)
            for word, audios in self.dataset.word_to_audios.items()
            for audio in audios
        ]
        self.word_audio = sorted(self.word_audio, key=lambda x: x[0])

        if to_shuffle and split == "train":
            random.shuffle(self.word_audio)

    def __getitem__(self, i):
        # worker_info = torch.utils.data.get_worker_info()
        # print("worker:", worker_info.id)
        # print("index: ", i)
        # print()

        def sample_neg(data, word):
            words = set(self.dataset.words_seen) - set([word])
            words = random.choices(list(words), k=self.n_neg)
            return [random.choice(data[word]) for word in words]

        word, audio_pos = self.word_audio[i]
        images_pos = random.choices(self.dataset.word_to_images[word], k=self.n_pos)
        audios_pos = random.choices(self.dataset.word_to_audios[word], k=self.n_pos - 1)
        audios_pos = [audio_pos] + audios_pos

        data_pos = [
            {
                "image": load_image(image_name),
                "audio": load_audio(audio_name),
                "label": 1,
            }
            for image_name, audio_name in zip(images_pos, audios_pos)
        ]

        images_neg = sample_neg(self.dataset.word_to_images, word)
        audios_neg = sample_neg(self.dataset.word_to_audios, word)

        data_neg = [
            {
                "image": load_image(image_name),
                "audio": load_audio(audio_name),
                "label": 0,
            }
            for image_name, audio_name in zip(images_neg, audios_neg)
        ]

        return default_collate(data_pos + data_neg)

    def __len__(self):
        return len(self.word_audio)


class PairedTestDataset(Dataset):
    def __init__(self, test_name):
        assert test_name in {"familiar-familiar", "novel-familiar"}
        super(PairedTestDataset).__init__()

        with open(f"mymme/data/filelists/{test_name}-test.json", "r") as f:
            self.data_pairs = json.load(f)

    def __getitem__(self, index: int):
        datum = self.data_pairs[index]
        assert datum["audio"]["word-en"] == datum["image-pos"]["word-en"]
        assert datum["audio"]["word-en"] != datum["image-neg"]["word-en"]
        return {
            "audio": load_audio(datum["audio"]),
            "image-pos": load_image(datum["image-pos"]),
            "image-neg": load_image(datum["image-neg"]),
        }

    def __len__(self):
        return len(self.data_pairs)


def setup_data(*, num_workers, batch_size, **dataset_kwargs):
    train_dataset = PairedMEDataset(split="train", **dataset_kwargs)
    valid_dataset = PairedMEDataset(split="valid", **dataset_kwargs)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return train_dataloader, valid_dataloader


def setup_data_paired_test(*, num_workers, batch_size):
    dataset_ff = PairedTestDataset("familiar-familiar")
    dataset_nf = PairedTestDataset("novel-familiar")

    dataloader_ff = DataLoader(
        dataset_ff,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    dataloader_nf = DataLoader(
        dataset_nf,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return dataloader_ff, dataloader_nf


def my_collate_fn(batch):
    audios = pad_sequence([datum["audio"] for datum in batch], batch_first=True)
    rest = [dissoc(datum, "audio") for datum in batch]
    rest = default_collate(rest)
    return {"audio": audios, **rest}


if __name__ == "__main__":
    num_pos = 4
    num_neg = 12
    dataset = PairedMEDataset(
        split="train",
        langs=("english",),
        num_pos=num_pos,
        num_neg=num_neg,
        # num_word_repeats=5,
    )
    dataloader = DataLoader(dataset, num_workers=4)
    import pdb

    pdb.set_trace()
    for batch in dataloader:
        # print(batch["image"][0, 0, :3, :3])
        pdb.set_trace()
