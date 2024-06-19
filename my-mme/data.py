from typing import Literal

import random
import json

import torch

from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.sampler import SubsetRandomSampler

from toolz import concat
from utils import read_file


Split = Literal["train", "valid", "test"]


class MEDataset:
    def __init__(self, split: Split):
        self.split = split
        self.words_seen = read_file("data/words-seen.txt")
        self.words_unseen = read_file("data/words-unseen.txt")

    def load_audio(self, name: str) -> torch.Tensor:
        pass

    def load_image(self, name: str) -> torch.Tensor:
        pass


class PairedMEDataset(IterableDataset):
    def __init__(self, split: Split, n_pos: int, n_neg: int):
        super(PairedMEDataset).__init__()
        self.dataset = MEDataset(split)
        self.n_pos = n_pos
        self.n_neg = n_neg

        assert split in ("train", "valid")

        worker_info = torch.utils.data.get_worker_info()
        seed = worker_info.id
        random.seed(seed)
        torch.manual_seed(seed)
        # np.random.seed(worker_info.id)

    def __iter__(self):
        while True:
            word = random.choice(self.dataset.words)

            images_pos = random.sample(self.dataset.images[word], self.n_pos)
            audios_pos = random.sample(self.dataset.audios[word], self.n_pos)

            for image_name, audio_name in zip(images_pos, audios_pos):
                yield {
                    "image": self.dataset.load_image(image_name),
                    "audio": self.dataset.load_audio(audio_name),
                    "label": 1,
                }

            images_neg = concat(self.dataset.images[w] for w in self.dataset.words if w != word)
            images_neg = random.sample(images_neg, self.n_neg)

            audios_neg = concat(self.dataset.audios[w] for w in self.dataset.words if w != word)
            audios_neg = random.sample(audios_neg, self.n_neg)

            for image_name, audio_name in zip(images_neg, audios_neg):
                yield {
                    "image": self.dataset.load_image(image_name),
                    "audio": self.dataset.load_audio(audio_name),
                    "label": 0,
                }



def setup_data(config):
    train_dataset = PairedMEDataset(split="train")
    valid_dataset = PairedMEDataset(split="valid")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size_train"],
        num_workers=config["num_workers"],
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size_valid"],
        num_workers=config["num_workers"],
    )

    return train_dataloader, valid_dataloader