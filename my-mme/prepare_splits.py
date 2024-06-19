import json
import random

from itertools import chain, combinations, groupby, product
from toolz import first, interleave

from sklearn.model_selection import train_test_split
from torchaudio.datasets import VoxCeleb1Identification


import pdb; pdb.set_trace()
seed = 42
random.seed(42)

dataset = VoxCeleb1Identification(
    subset="test",
    root="data/voxceleb1",
    download=False,
)

num_utterances = len(dataset)

speaker_ids = [dataset.get_metadata(i)[2] for i in range(num_utterances)]
speaker_ids_unique = list(sorted(set(speaker_ids)))

speakers_ids_tr, speakers_ids_te = train_test_split(
    speaker_ids_unique,
    test_size=251,
    random_state=seed,
)

utterances_ids_tr = [
    i for i in range(num_utterances) if dataset.get_metadata(i)[2] in speakers_ids_tr
]
utterances_ids_te = [
    i for i in range(num_utterances) if dataset.get_metadata(i)[2] in speakers_ids_te
]

utterances_ids_tr, utterances_ids_va = train_test_split(
    utterances_ids_tr,
    test_size=0.1,
    random_state=seed,
)

assert set(utterances_ids_tr) & set(utterances_ids_va) == set()
assert set(utterances_ids_tr) & set(utterances_ids_te) == set()
assert set(utterances_ids_va) & set(utterances_ids_te) == set()

print(len(utterances_ids_tr))
print(len(utterances_ids_va))
print(len(utterances_ids_te))

data = {
    "train": sorted(utterances_ids_tr),
    "valid": sorted(utterances_ids_va),
    "test": sorted(utterances_ids_te),
}

with open("data/voxceleb1/a3/splits-samples.json", "w") as f:
    json.dump(data, f)


speakers = {
    "train": speakers_ids_tr,
    "valid": speakers_ids_tr,
    "test": speakers_ids_te,
}

with open("data/voxceleb1/a3/splits-speakers.json", "w") as f:
    json.dump(speakers, f)


speaker_id_and_utterance_id = zip(speaker_ids, range(num_utterances))
speaker_id_and_utterance_id = sorted(speaker_id_and_utterance_id, key=first)
speaker_id_to_utterance_ids = {
    speaker_id: [utterance_id for _, utterance_id in group]
    for speaker_id, group in groupby(speaker_id_and_utterance_id, key=first)
}


pairs_pos = chain.from_iterable(
    combinations(speaker_id_to_utterance_ids[speaker_id], 2)
    for speaker_id in speakers_ids_te
)

pairs_neg = chain.from_iterable(
    product(
        speaker_id_to_utterance_ids[s],
        speaker_id_to_utterance_ids[t],
    )
    for s, t in combinations(speakers_ids_te, 2)
)


def prepare_pair(label, pair):
    s, t = pair
    path1 = dataset.get_metadata(s)[0]
    path2 = dataset.get_metadata(t)[0]
    return {
        "label": label, 
        "path1": path1,
        "path2": path2,
    }


pairs_pos = random.sample(list(pairs_pos), 1000)
pairs_neg = random.sample(list(pairs_neg), 1000)

pairs_pos = [prepare_pair(1, pair) for pair in pairs_pos]
pairs_neg = [prepare_pair(0, pair) for pair in pairs_neg]

pairs = list(interleave([pairs_pos, pairs_neg]))

with open("data/voxceleb1/a3/test-pairs.json", "w") as f:
    json.dump(pairs, f)
