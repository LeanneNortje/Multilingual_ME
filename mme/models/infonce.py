import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


def infonce(anchor, positives_1, positives_2, negatives_1, negatives_2):
    # print(anchor.size(), positives_1.size(), positives_2.size(), negatives_1.size(), negatives_2.size())
    b, n_pos, _ = positives_1.shape
    _, n_neg, _ = negatives_1.shape
    logits = torch.cat(
        [
            torch.cat([anchor, negatives_1], dim=1),
            torch.cat([anchor, negatives_2], dim=1),
            torch.cat([positives_1.transpose(1, 2), negatives_1.repeat(1, 1, n_pos)], dim=1),
            torch.cat([positives_2.transpose(1, 2), negatives_2.repeat(1, 1, n_pos)], dim=1),
        ],
        dim=2,
    )
    logits = logits.transpose(1, 2).reshape(b * (2 + 2 * n_pos), (1 + n_neg))
    # logits = logits.transpose(1, 2).reshape(b * 2, (1 + n_neg))
    labels = torch.zeros(len(logits), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)


# B = 16
# N_POS = 5
# N_NEG = 11

# anchor = torch.randn(B, 1, 1)
# positives_1 = torch.randn(B, N_POS, 1)
# positives_2 = torch.randn(B, N_POS, 1)
# negatives_1 = torch.randn(B, N_NEG, 1)
# negatives_2 = torch.randn(B, N_NEG, 1)

# print(infonce(anchor, positives_1, positives_2, negatives_1, negatives_2))
# print()

# print("Good similarity for positivies")
# print(infonce(5 + anchor, 5 + positives_1, 5 + positives_2, negatives_1, negatives_2))
