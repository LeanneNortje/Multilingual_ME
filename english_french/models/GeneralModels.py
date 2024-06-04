#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoringAttentionModule(nn.Module):
    def __init__(self, args):
        super(ScoringAttentionModule, self).__init__()

    def forward(self, embedding_1, embedding_2):

        att = torch.bmm(embedding_1.transpose(1, 2), embedding_2)#.squeeze(2)# / (torch.norm(aud_em, dim=1) * torch.norm(image_embedding, dim=1))
        s, _ = att.max(dim=-1)
        s, _ = s.max(dim=-1)
        return s.unsqueeze(-1) 

    def score(self, embedding_1, embedding_2):
        scores = [] #torch.zeros((audio_embeddings.size(0), image_embedding.size(0)), device=audio_embeddings.device)
        for i in range(embedding_1.size(0)):
            
            att = torch.bmm(embedding_1[i, :, :].unsqueeze(0).transpose(1, 2), embedding_2[i, :, :].unsqueeze(0))#.squeeze(2)# / (torch.norm(aud_em, dim=1) * torch.norm(im, dim=1))
            s, _ = att.max(dim=-1)
            s, _ = s.max(dim=-1)
            scores.append(s.unsqueeze(-1))
        scores = torch.cat(scores, dim=0)
        return scores #self.sig(scores)

    def attention_scores(self, image, audio):

        att = torch.bmm(audio.transpose(1, 2), image)#.squeeze(2)# / (torch.norm(aud_em, dim=1) * torch.norm(im, dim=1)
        ind = att.argmax().item()
        embedding = image[:, :, ind].squeeze()
        
        return embedding #self.sig(scores)

    def get_attention(self, image, audio):
        return torch.bmm(audio.transpose(1, 2), image)

    def one_to_many_score(self, embeddings, embedding_1):

        scores = []
        for i in range(embeddings.size(0)):
            att = torch.bmm(embedding_1.transpose(1, 2), embeddings[i, :, :].unsqueeze(0))#.squeeze(2)# / (torch.norm(aud_em, dim=1) * torch.norm(im, dim=1)
            s, _ = att.max(dim=-1)
            s, _ = s.max(dim=-1)
            scores.append(s.unsqueeze(-1))
        scores = torch.cat(scores, dim=1)
        return scores #self.sig(scores)
        
class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()

        self.embedding_dim = args["audio_model"]["embedding_dim"]
        self.margin = args["margin"]
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.criterion = nn.MSELoss()

    def forward(self, anchor, positives_1, positives_2, negatives_1, negatives_2):
        # print(anchor.size(), positives_1.size(), positives_2.size(), negatives_1.size(), negatives_2.size())
        N = anchor.size(0)
        sim = [anchor, positives_1, positives_2, negatives_1, negatives_2]
        # if base_negatives is not None: sim.append(base_negatives)
        sim = torch.cat(sim, dim=1)
        labels = []
        labels.append(100*torch.ones((N, anchor.size(1)), device=anchor.device))
        labels.append(100*torch.ones((N, positives_1.size(1)), device=anchor.device))
        labels.append(100*torch.ones((N, positives_2.size(1)), device=anchor.device))
        labels.append(0*torch.ones((N, negatives_1.size(1)), device=anchor.device))
        labels.append(0*torch.ones((N, negatives_2.size(1)), device=anchor.device))
        # if base_negatives is not None: labels.append(0*torch.ones((N, base_negatives.size(1)), device=anchor.device))
        labels = torch.cat(labels, dim=1)
        loss = self.criterion(sim, labels)

        return loss