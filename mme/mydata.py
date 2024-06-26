from itertools import groupby
from typing import Literal, Tuple

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

    return read_file("../mymme/data/concepts.txt", parse_line)


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

        self.words_seen = read_file("../mymme/data/words-seen.txt")
        self.words_unseen = read_file("../mymme/data/words-unseen.txt")

        image_files = read_json(f"../mymme/data/filelists/image-{split}.json")
        audio_files = read_json(f"../mymme/data/filelists/audio-{split}.json")
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
        self.get_word = self.get_word_train if split == "train" else self.get_word_valid

    def get_word_train(self):
        while True:
            # yield random.choice(self.dataset.words_seen)
            yield "bear"

    def get_word_valid(self):
        return self.dataset.words_seen

    # def __iter__(self):
    #     # worker_info = torch.utils.data.get_worker_info()
    #     # print(worker_info.id)

    #     while True:
    #         for i, word in enumerate(self.dataset.words_seen):
    #             image = random.choice(self.dataset.word_to_images[word])
    #             audio = random.choice(self.dataset.word_to_audios[word])
    #             # image = self.dataset.word_to_images[word][0]
    #             # audio = self.dataset.word_to_audios[word][0]

    #             yield {
    #                 "image": load_image(image),
    #                 "audio": load_audio(audio),
    #                 "label": i,
    #             }

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        # print(worker_info.id)

        def sample_neg(data, word):
            words = set(self.dataset.words_seen) - set([word])
            words = random.choices(list(words), k=self.n_neg)
            return [random.choice(data[word]) for word in words]
            # return data["boat"][: self.n_neg]

        for word in self.get_word():
            images_pos = random.choices(self.dataset.word_to_images[word], k=self.n_pos)
            audios_pos = random.choices(self.dataset.word_to_audios[word], k=self.n_pos)

            # images_pos = self.dataset.word_to_images[word][: self.n_pos]
            # audios_pos = self.dataset.word_to_audios[word][: self.n_pos]

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


def setup_data(*, num_pos, num_neg, num_workers):
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
        num_workers=num_workers,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=1,
    )

    return train_dataloader, valid_dataloader


class LeanneDataset(Dataset):
    def __init__(self, split):
        super(PairedMEDataset).__init__()

        assert split in ("train", "valid")
        langs = ("english", )
        self.dataset = MEDataset(split, langs)
        self.word_audio = [
            (word, audio)
            for word in sorted(self.dataset.words_seen)
            for audio in self.dataset.word_to_audios[word]
        ]
 
    def __getitem__(self, i):

        def sample_neg(data, word):
            words = set(self.dataset.words_seen) - set([word])
            word = random.choice(list(words))
            return random.choice(data[word])

        def load_pos(word):
            audio = random.choice(self.dataset.word_to_audios[word])
            audio = load_audio(audio)
            image = random.choice(self.dataset.word_to_images[word])
            image = load_image(image)
            return {
                "pos_image": image,
                "pos_english": audio,
                "pos_french": audio,
                "pos_dutch": audio,
            }

        def load_neg(word):
            audio = sample_neg(self.dataset.word_to_audios, word)
            audio = load_audio(audio)
            image = sample_neg(self.dataset.word_to_images, word)
            image = load_image(image)
            return {
                "neg_image": image,
                "neg_english": audio,
                "neg_french": audio,
                "neg_dutch": audio,
            }

        word, audio = self.word_audio[i]

        audio = load_audio(audio)
        image = random.choice(self.dataset.word_to_images[word])
        image = load_image(image)

        positives = [load_pos(word) for _ in range(5)]
        negatives = [load_neg(word) for _ in range(11)]

        # import pdb; pdb.set_trace()

        return {
            "image": image,
            "english_feat": audio,
            "french_feat": audio,
            "dutch_feat": audio,
            "positives": positives,
            "negatives": negatives,
        }

    def __len__(self):
        return len(self.word_audio)


def my_collate_fn(batch):
    audios = pad_sequence([datum["audio"] for datum in batch], batch_first=True)
    est = [dissoc(datum, "audio") for datum in batch]
    rest = default_collate(rest)
    return {"audio": audios, **rest}


if __name__ == "__main__":
    n_pos = 4
    n_neg = 12
    batch_size = n_pos + n_neg
    dataset = PairedMEDataset(
        split="train", langs=("english",), n_pos=n_pos, n_neg=n_neg
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    for batch in dataloader:
        print(batch)
        pdb.set_trace()
