from typing import List, Dict
from pathlib import Path

import json
import pdb
import random
import numpy as np

from toolz import first

from mymme.data import load_dictionary, MEDataset


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
    data1 = [tuple(datum.values()) for datum in data]
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


def prepare_test_filelist():
    def unpack_datum(datum):
        word, image_path, audio_path, image_source = datum
        entry_image = {
            "name": image_path.stem,
            "word-en": word,
            "source": image_source,
        }
        entry_audio = {
            "lang": "english",
            "name": audio_path.stem,
            "word-en": word,
        }
        return entry_audio, entry_image

    path = "mme/data/episodes.npz"
    data = np.load(path, allow_pickle=True)["episodes"].item()

    data_out = [
        unpack_datum(datum4)
        for datum1 in data.values()
        for datum2 in datum1.values()
        for datum3 in datum2.values()
        for datum4 in datum3
    ]

    data_audio, data_image = zip(*data_out)
    data_audio = deduplicate(data_audio)
    data_image = deduplicate(data_image)

    # Sanity check (seems good!)

    # def extract_names(data):
    #     data1 = data.item().values()
    #     return [datum2.stem for datum1 in data1 for datum2 in datum1]

    # path = "mme/results/files/episode_data.npz"
    # data = np.load(path, allow_pickle=True)

    # audio_names = extract_names(data["audio_1"]) + extract_names(data["audio_2"])
    # audio_names = set(audio_names)
    # image_names = extract_names(data["image_1"]) + extract_names(data["image_2"])
    # image_names = set(image_names)

    # print(len(audio_names), len(image_names))
    # print(len(data_audio), len(data_image))

    # assert audio_names == set([d["name"] for d in data_audio])
    # assert image_names == set([d["name"] for d in data_image])
    # pdb.set_trace()

    save_data(data_audio, "audio", "test")
    save_data(data_image, "image", "test")


def prepare_familiar_familiar(split, num_word_repeat):
    # split = "test"
    assert split in ("train", "valid", "test")
    langs = ("english",)
    dataset = MEDataset(split, langs)

    def sample_neg_train_valid(data, image_pos):
        word = image_pos["word-en"]
        words = set(dataset.words_seen) - set([word])
        word_other = random.choice(list(words))
        return random.choice(data[word_other])

    def sample_neg_test(data, image_pos):
        data1 = [
            datum
            for word in dataset.words_seen
            if word != image_pos["word-en"]
            for datum in data[word]
            if datum["source"] == image_pos["source"]
        ]
        return random.choice(data1)

    SAMPLE_NEGS = {
        "train": sample_neg_train_valid,
        "valid": sample_neg_train_valid,
        "test": sample_neg_test,
    }
    sample_neg = SAMPLE_NEGS[split]

    def sample_pair(word):
        audio = random.choice(dataset.word_to_audios[word])
        image_pos = random.choice(dataset.word_to_images[word])
        image_neg = sample_neg(dataset.word_to_images, image_pos)
        return {
            "audio": audio,
            "image-pos": image_pos,
            "image-neg": image_neg,
        }

    data = [
        sample_pair(word) for word in dataset.words_seen for _ in range(num_word_repeat)
    ]

    with open(f"mymme/data/filelists/familiar-familiar-{split}.json", "w") as f:
        json.dump(data, f, indent=2)


def prepare_novel_familiar(num_word_repeat):
    split = "test"
    langs = ("english",)
    dataset = MEDataset(split, langs)

    def sample_neg(data, image_pos):
        data1 = [
            datum
            for word in dataset.words_seen
            for datum in data[word]
            if datum["source"] == image_pos["source"]
        ]
        datum = random.choice(data1)
        assert datum["word-en"] != image_pos["word-en"]
        return datum

    def sample_pair(word):
        audio = random.choice(dataset.word_to_audios[word])
        image_pos = random.choice(dataset.word_to_images[word])
        image_neg = sample_neg(dataset.word_to_images, image_pos)
        return {
            "audio": audio,
            "image-pos": image_pos,
            "image-neg": image_neg,
        }

    data = [
        sample_pair(word)
        for word in dataset.words_unseen
        for _ in range(num_word_repeat)
    ]

    with open(f"mymme/data/filelists/novel-familiar-{split}.json", "w") as f:
        json.dump(data, f, indent=2)


def extract_filelists_from_leanne():
    dataset = MEDataset("test", langs=("english",))

    path = "mme/results/files/episode_data.npz"
    data = np.load(path, allow_pickle=True)

    audio = data["audio_1"].item()
    audio_words = data["audio_labels_1"].item()

    image_pos = data["image_1"].item()
    image_pos_words = data["image_labels_1"].item()

    image_neg = data["image_2"].item()
    image_neg_words = data["image_labels_2"].item()

    episodes = list(range(1000))
    episodes = random.sample(episodes, 30)

    data = [
        {
            "audio": {
                "name": audio[e][i].stem,
                "word-en": audio_words[e][i],
                "lang": "english",
            },
            "image-pos": {
                "name": image_pos[e][i].stem,
                "word-en": image_pos_words[e][i],
            },
            "image-neg": {
                "name": image_neg[e][i].stem,
                "word-en": image_neg_words[e][i],
            },
        }
        for e in episodes
        for i in range(33)
    ]

    data_ff = [
        datum
        for datum in data
        if datum["audio"]["word-en"] in dataset.words_seen
        and datum["image-pos"]["word-en"] in dataset.words_seen
        and datum["image-neg"]["word-en"] in dataset.words_seen
    ]

    data_nf = [
        datum
        for datum in data
        if datum["audio"]["word-en"] in dataset.words_unseen
        and datum["image-pos"]["word-en"] in dataset.words_unseen
        and datum["image-neg"]["word-en"] in dataset.words_seen
    ]

    with open("mymme/data/filelists/leanne-familiar-familiar-test.json", "w") as f:
        json.dump(data_ff, f, indent=2)

    with open("mymme/data/filelists/leanne-novel-familiar-test.json", "w") as f:
        json.dump(data_nf, f, indent=2)


if __name__ == "__main__":
    # prepare_audio_filelist("train")
    # prepare_audio_filelist("valid")
    # prepare_image_filelist("train")
    # prepare_image_filelist("valid")
    # prepare_familiar_familiar("test", 10)
    # prepare_novel_familiar(10)
    extract_filelists_from_leanne()
