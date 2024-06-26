import pdb
import random

from pathlib import Path

import streamlit as st
import torch

from mymme.train import CONFIGS, setup_model
from mymme.data import MEDataset, get_audio_path, get_image_path, load_audio, load_image


def get_best_checkpoint(output_dir: Path) -> Path:
    def get_neg_loss(file):
        *_, neg_loss = file.stem.split("=")
        return float(neg_loss)

    folder = output_dir / "checkpoints"
    files = folder.iterdir()
    file = max(files, key=get_neg_loss)
    print(file)
    return file


def load_model(config_name, config):
    model = setup_model(**config["model"])
    folder = Path("output") / config_name
    state = torch.load(get_best_checkpoint(folder))
    model.load_state_dict(state)
    model.to(device=config["device"])
    model.eval()
    return model


dataset = MEDataset("train", langs=("english",))

config_name = "00"
config = CONFIGS[config_name]
model = load_model(config_name, config)

# audio_names = [
#     "bear_6794-73984-0066",
#     "bear_2920-156230-0017",
#     "bear_5727-47030-0033",
#     "bear_2405-182390-0025",
# ]
# 
# image_names = [
#     "bear_n02133161_1737",
#     "bear_n02132136_9478",
#     "bear_n02134084_5807",
#     "bear_n02132136_6388",
# ]

# for audio_name, image_name in zip(audio_names, image_names):
#     audio = load_audio({"name": audio_name, "lang": "english"})
#     audio = audio.unsqueeze(0)
#     image = load_image({"name": image_name})
#     image = image.unsqueeze(0)
#     with torch.no_grad():
#         pdb.set_trace()
#         score = model.score(audio, image, "pair")
#         print(score)


for _ in range(10):
    word = "bear"
    # word = random.choice(dataset.words_seen)
    words_other = set(dataset.words_seen) - set([word])
    word_neg = random.choice(list(words_other))

    datum_audio = random.choice(dataset.word_to_audios[word])
    datum_image_pos = random.choice(dataset.word_to_images[word])
    datum_image_neg = random.choice(dataset.word_to_images[word_neg])

    audio = load_audio(datum_audio)
    audio = audio.unsqueeze(0)
    image_pos = load_image(datum_image_pos)
    image_pos = image_pos.unsqueeze(0)
    image_neg = load_image(datum_image_neg)
    image_neg = image_neg.unsqueeze(0)

    with torch.no_grad():
        score_pos = model.score(audio, image_pos, "pair")
        score_neg = model.score(audio, image_neg, "pair")

    st.write(datum_audio)
    st.audio(get_audio_path(datum_audio))
    col1, col2 = st.columns(2)
    col1.write(score_pos)
    col1.write(datum_image_pos)
    col1.image(get_image_path(datum_image_pos))
    col2.write(score_neg)
    col2.write(datum_image_neg)
    col2.image(get_image_path(datum_image_neg))
    st.markdown("---")
