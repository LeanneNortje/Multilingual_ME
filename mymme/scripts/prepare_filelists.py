from typing import List, Dict
from pathlib import Path

import json
import pdb
import random
import numpy as np

from toolz import first

from mymme.data import load_dictionary


random.seed(42)


DATA_DIR = Path("./data")
WORD_DICT = load_dictionary()


def load_data(split):
    SPLITS = {
        "train": "train",
        "valid": "val",
    }
    split_short = SPLITS[split]

    data = np.load(DATA_DIR / f"{split_short}_lookup.npz", allow_pickle=True)
    return data["lookup"].item()


def save_data(data, modality, split):
    data = deduplicate(data)
    path_out = f"mymme/data/filelists/{modality}-{split}.json"
    with open(path_out, "w") as f:
        json.dump(data, f, indent=2)


def deduplicate(data: List[Dict]) -> List[Dict]:
    keys = data[0].keys()
    data1 = [datum.values() for datum in data]
    data1 = set(data1)
    return [dict(zip(keys, datum)) for datum in data1]


def extract_word_en(lang, audio_name: str) -> str:
    word = audio_name.split("_")[0]
    entry = first(e for e in WORD_DICT if e[lang] == word)
    return entry["english"]


def prepare_audio_filelist(split):
    data = load_data(split)
    data_audio = [
        {
            "lang": lang,
            "name": audio_name.stem,
            "word-en": extract_word_en(lang, audio_name.stem),
        }
        for _, data1 in data.items()
        for lang, data2 in data1.items()
        if lang != "images"
        for _, audio_names in data2.items()
        for audio_name in audio_names
    ]
    save_data(data_audio, "audio", split)


def prepare_image_filelist(split):
    data = load_data(split)
    data_image = [
        {
            "name": image_name.stem,
            "word-en": image_name.stem.split("_")[0],
        }
        for _, data1 in data.items()
        for lang, data2 in data1.items()
        if lang == "images"
        for _, image_names in data2.items()
        for image_name in image_names
    ]
    save_data(data_image, "image", split)


def prepare_pairs_filelist(split, langs, num_word_repeat):
    from mymme.data import MEDataset

    assert split in ["train", "valid"]
    dataset = MEDataset(split, langs)

    def sample_neg(data, word):
        words = set(dataset.words_seen) - set([word])
        word = random.choice(list(words))
        return random.choice(data[word])

    def sample_pair(word):
        audio = random.choice(dataset.word_to_audios[word])
        image_pos = random.choice(dataset.word_to_images[word])
        image_neg = sample_neg(dataset.word_to_images, word)
        return {
            "audio": audio,
            "image-pos": image_pos,
            "image-neg": image_neg,
        }

    data = [
        sample_pair(word)
        for word in dataset.words_seen
        for _ in range(num_word_repeat)
    ]

    with open(f"mymme/data/filelists/pairs-{split}.json", "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    # prepare_audio_filelist("train")
    # prepare_audio_filelist("valid")
    # prepare_image_filelist("train")
    # prepare_image_filelist("valid")
    prepare_pairs_filelist("valid", ("english", ), 10)
