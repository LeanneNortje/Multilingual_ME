import json
import pdb
import random

from functools import partial
from pathlib import Path
from tqdm import tqdm

import streamlit as st
import torch

from torch.utils.data import DataLoader

from mymme.train import CONFIGS, setup_model, UtilsPairedTest
from mymme.data import (
    PairedMEDataset,
    PairedTestDataset,
    collate_with_audio,
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
            "english_model": drop_prefix(state["english_model"], "module"),
            # "english_model": drop_prefix(state["audio_model"], "module"),
            "image_model": drop_prefix(state["image_model"], "module"),
        }

    model = setup_model(**config["model"])
    model.image_enc.load_state_dict(state["image_model"])
    model.audio_enc.load_state_dict(state["english_model"])
    model.eval()
    return model


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


def score_pair(model, datum, device):
    audio = load_audio(datum["audio"])
    audio = audio.unsqueeze(0).to(device)
    image_pos = load_image(datum["image-pos"])
    image_pos = image_pos.unsqueeze(0).to(device)
    image_neg = load_image(datum["image-neg"])
    image_neg = image_neg.unsqueeze(0).to(device)

    assert datum["audio"]["word-en"] == datum["image-pos"]["word-en"]
    assert datum["audio"]["word-en"] != datum["image-neg"]["word-en"]

    audio_length = torch.tensor([audio.size(2)])
    audio_length = audio_length.to(device)

    with torch.no_grad():
        score_pos = model.score(audio, audio_length, image_pos, "pair")
        score_neg = model.score(audio, audio_length, image_neg, "pair")

    is_correct = score_pos > score_neg
    is_correct = bool(is_correct)

    return {
        "score-pos": score_pos,
        "score-neg": score_neg,
        "is-correct": is_correct,
        **datum,
    }


def evaluate_model_batched(test_name, model, device):
    dataset = PairedTestDataset(test_name)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=4,
        collate_fn=collate_with_audio,
    )
    model_fn = lambda model, inputs: model.predict_paired_test(*inputs)
    prepare_batch_fn = UtilsPairedTest.prepare_batch_fn
    with torch.no_grad():
        preds = [
            model_fn(
                model,
                prepare_batch_fn(batch, device, non_blocking=True)[0],
            )
            for batch in tqdm(dataloader)
        ]
    preds = torch.cat(preds, dim=0)
    is_correct = (preds[:, 0] > preds[:, 1]).float()
    return 100 * torch.mean(is_correct).item()


def evaluate_model(test_name, model, device):
    with open(f"mymme/data/filelists/{test_name}-test.json", "r") as f:
        data_pairs = json.load(f)

    results = [score_pair(model, datum, device) for datum in tqdm(data_pairs)]
    num_correct = sum(result["is-correct"] for result in results)
    num_total = len(results)
    return 100 * num_correct / num_total


# config_name = "04"
# config = CONFIGS[config_name]
# # model = load_model(config_name, config)
# model = load_model_leanne(
#     config_name,
#     config,
#     path="/home/doneata/data/mme/checkpoints/english/model_metadata/b4b77a981b/1/models/best_ckpt.pt",
#     # path="mme/model_metadata/a79eb05d20/2/models/epoch_3.pt",
#     # path="trilingual_no_language_links/model_metadata/0a0057c11d/2/models/epoch_2.pt",  # works
#     # path="english/model_metadata/baseline-shuffle/1/models/epoch_1.pt",  # doesn't work
#     to_drop_prefix=True,
# )

MODELS = {k: partial(load_model, k, config) for k, config in CONFIGS.items()}

for i in range(1, 4):
    MODELS[f"b4b77a981b/{i}"] = partial(
        load_model_leanne,
        config_name="04",
        config=CONFIGS["04"],
        path=f"/home/doneata/data/mme/checkpoints/english/model_metadata/b4b77a981b/{i}/models/best_ckpt.pt",
        to_drop_prefix=True,
    )


if __name__ == "__main__":
    with st.sidebar:
        model_name = st.selectbox("Model:", list(MODELS.keys()))
        test_name = st.selectbox(
            "Test:",
            [
                "familiar-familiar",
                "novel-familiar",
                "leanne-familiar-familiar",
                "leanne-novel-familiar",
            ],
        )

    DEVICE = "cuda"
    model = MODELS[model_name]()
    model.to(DEVICE)

    with open(f"mymme/data/filelists/{test_name}-test.json") as f:
        data_pairs = json.load(f)

    results = [score_pair(model, datum, DEVICE) for datum in tqdm(data_pairs)]
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
