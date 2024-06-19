from typing import List, Dict
from pathlib import Path

import pdb
import numpy as np

from toolz import first

from data import load_dictionary


def deduplicate(data: List[Dict]) -> List[Dict]:
    keys = data[0].keys()
    data1 = [datum.values() for datum in data]
    data1 = set(data1)
    return [dict(zip(keys, datum)) for datum in data1]


word_dict = load_dictionary()


def extract_word_en(lang, audio_name: str) -> str:
    word = audio_name.split("_")[0]
    entry = first(e for e in word_dict if e[lang] == word)
    return entry["en"]
    

DATA_DIR = Path("../data")
data = np.load(DATA_DIR / "train_lookup.npz", allow_pickle=True)
data = data['lookup'].item()

data_audio = [
  {
    "lang": lang,
    "audio": audio_name.stem,
    "word-en": extract_word_en(lang, audio_name.stem),
  }
  for _, data1 in data.items()
  for lang, data2 in data1.items()
  if lang != "images"
  for _, audio_names in data2.items()
  for audio_name in audio_names
]

data_audio = deduplicate(data_audio)
print(len(data_audio))
print(data_audio[:5])