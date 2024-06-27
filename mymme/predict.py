import json
import pdb
import random

from pathlib import Path
from tqdm import tqdm

import streamlit as st
import torch

from mymme.train import CONFIGS, setup_model
from mymme.data import (
    MEDataset,
    PairedMEDataset,
    get_audio_path,
    get_image_path,
    load_audio,
    load_image,
)


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
    model.eval()
    return model


def load_model_leanne(config_name, config, path, to_drop_prefix):
    def drop_prefix(state, prefix, sep="."):
        def drop1(s):
            fst, *rest = s.split(sep)
            assert fst == prefix
            return sep.join(rest)

        return {drop1(k): v for k, v in state.items()}

    state = torch.load(path)

    if to_drop_prefix:
        state = {
            # "english_model": drop_prefix(state["english_model"], "module"),
            "english_model": drop_prefix(state["audio_model"], "module"),
            "image_model": drop_prefix(state["image_model"], "module"),
        }

    model = setup_model(**config["model"])
    model.image_enc.load_state_dict(state["image_model"])
    model.audio_enc.load_state_dict(state["english_model"])
    model.eval()
    return model


DEVICE = "cuda"
config_name = "02"
config = CONFIGS[config_name]
# model = load_model(config_name, config)
model = load_model_leanne(
    config_name,
    config,
    # path="/home/doneata/data/mme/checkpoints/english/model_metadata/b4b77a981b/1/models/best_ckpt.pt",
    # path="mme/model_metadata/a79eb05d20/2/models/epoch_3.pt",
    # path="trilingual_no_language_links/model_metadata/0a0057c11d/2/models/epoch_2.pt",  # works
    path="english/model_metadata/baseline/1/models/epoch_5.pt",  # doesn't work
    to_drop_prefix=True,
)
model.to(DEVICE)


# def get_scores(batch):
#     audio = batch["audio"]
#     image = batch["image"]
#     with torch.no_grad():
#         sim = model.score(audio, image, "cross")
#     return sim[0]


# num_words = 13
# num_pos = 1
# num_neg = 5
# dataset_paired = PairedMEDataset(
#     "valid",
#     langs=("english",),
#     num_pos=num_pos,
#     num_neg=num_neg,
#     # num_word_repeats=1,
# )
# scores = [get_scores(dataset_paired[i]) for i in range(num_words)]
# scores = torch.stack(scores)
# indices = scores.argsort(dim=1, descending=True)[:, :num_pos]
# accuracy = (indices < num_pos).float().mean()
# st.write(indices)
# st.markdown("Accurcay: {:.2f}".format(100 * accuracy))
# st.markdown("---")

# num_correct = 0
# num_samples = 10
# dataset = MEDataset("valid", langs=("english",))


with open("mymme/data/filelists/pairs-valid.json") as f:
    data_pairs = json.load(f)


def score_pair(datum):
    audio = load_audio(datum["audio"])
    audio = audio.unsqueeze(0).to(DEVICE)
    image_pos = load_image(datum["image-pos"])
    image_pos = image_pos.unsqueeze(0).to(DEVICE)
    image_neg = load_image(datum["image-neg"])
    image_neg = image_neg.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        score_pos = model.score(audio, image_pos, "pair")
        score_neg = model.score(audio, image_neg, "pair")

    is_correct = score_pos > score_neg
    is_correct = bool(is_correct)

    return {
        "score-pos": score_pos,
        "score-neg": score_neg,
        "is-correct": is_correct,
        **datum,
    }


results = [score_pair(datum) for datum in tqdm(data_pairs)]

num_correct = sum(result["is-correct"] for result in results)
num_total = len(results)
accuracy = 100 * num_correct / num_total

st.write(f"Accuracy: {accuracy:.2f}%")
st.markdown("---")


for result in results[:10]:
    word = result["audio"]["word-en"]
    is_correct = result["is-correct"]
    is_correct_str = "✓" if is_correct else "✗"

    st.markdown(f"### {word} · is correct: {is_correct_str}")
    st.write(result["audio"])
    st.audio(get_audio_path(result["audio"]))

    col1, col2 = st.columns(2)
    col1.write(result["score-pos"])
    col1.write(result["image-pos"])
    col1.image(get_image_path(result["image-pos"]))

    col2.write(result["score-neg"])
    col2.write(result["image-neg"])
    col2.image(get_image_path(result["image-neg"]))

    st.markdown("---")
