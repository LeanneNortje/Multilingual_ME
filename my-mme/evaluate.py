import json

from pathlib import Path

import click
import librosa
import torch

from omegaconf import OmegaConf
from sklearn.metrics import roc_curve
from tqdm import tqdm

from scipy.optimize import brentq
from scipy.interpolate import interp1d

from speechbrain.lobes.features import Fbank
from speechbrain.inference.speaker import EncoderClassifier

from model import setup_model


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
    model = setup_model(**config.model)
    folder = Path("output") / config_name
    state = torch.load(get_best_checkpoint(folder))
    model.load_state_dict(state)
    model.to(device=config.device)
    model.eval()
    return model


class SpeakerVerificationScorer:
    def score(self, audio1, audio2) -> float:
        raise NotImplementedError


class ECAPATDNNScorer(SpeakerVerificationScorer):
    def __init__(self, *args, **kwargs):
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda"},
        )
        # self.classifier.eval()
        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def score(self, audio1, audio2) -> float:
        with torch.no_grad():
            sim = self.similarity(
                self.classifier.encode_batch(audio1),
                self.classifier.encode_batch(audio2),
            )
        return sim.cpu().item()


class OurScorer(SpeakerVerificationScorer):
    def __init__(self, config_name):
        self.config = OmegaConf.load(f"configs/{config_name}.yaml")
        self.extract_features = Fbank(n_mels=self.config.features.n_mels)
        self.extract_features.to(self.config.device)
        self.model = load_model(config_name, self.config)
        self.speaker_to_features = {}
        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def score(self, audio1, audio2):
        def get_feature(audio):
            audio = audio.to(self.config.device)
            feature = self.extract_features(audio)
            feature = self.model.get_embedding(feature)
            feature = self.model.average_pooling(feature, None)
            return feature.unsqueeze(0)
        score = self.similarity(
            get_feature(audio1),
            get_feature(audio2)
        )
        return score.cpu().item()


def compute_eer(y_true, y_pred, pos_label):
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=pos_label)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return 100 * eer


SCORERS = {
    "ecapa-tdnn": ECAPATDNNScorer,
    "ours": OurScorer,
}


def load_audio(subpath):
    root = Path("data/voxceleb1/wav")
    path = root / subpath
    audio, _ = librosa.load(path, sr=16_000)
    return torch.tensor(audio).unsqueeze(0)


def evaluate(scorer_name, config_name):
    with open("data/voxceleb1/a3/test-pairs.json") as f:
        pairs = json.load(f)

    system = SCORERS[scorer_name](config_name)

    labels = [pair["label"] for pair in pairs]
    scores = [
        system.score(load_audio(pair["path1"]), load_audio(pair["path2"]))
        for pair in tqdm(pairs)
    ]

    # print(labels)
    # print(scores)
    print(compute_eer(labels, scores))


@click.command()
@click.option(
    "-s",
    "--scorer",
    "scorer_name",
    type=str,
    required=True,
)
@click.option(
    "-c",
    "--config",
    "config_name",
    type=str,
    required=True,
)
def main(scorer_name, config_name):
    evaluate(scorer_name, config_name)


if __name__ == "__main__":
    main()
