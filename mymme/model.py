import torch
import torch.nn as nn
from torchvision import alexnet


class AudioEncoder(nn.Module):
    def __init__(self, args):
        super(AudioEncoder, self).__init__()
        num_channels = args["acoustic_model"]["out_channels"]
        z_dim = args["audio_model"]["z_dim"]
        c_dim = args["audio_model"]["c_dim"]
        frame_dim = args["audio_config"]["target_length"]

        self.conv = nn.Conv1d(
            args["acoustic_model"]["in_channels"],
            num_channels,
            args["acoustic_model"]["kernel_size"],
            args["acoustic_model"]["stride"],
            args["acoustic_model"]["padding"],
            bias=False,
        )

        self.encoder = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # nn.InstanceNorm1d(num_channels),
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # nn.InstanceNorm1d(num_channels),
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # nn.InstanceNorm1d(num_channels),
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # nn.InstanceNorm1d(num_channels),
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, z_dim),
            # nn.InstanceNorm1d(z_dim),
        )
        self.rnn1 = nn.LSTM(z_dim, c_dim, batch_first=True)
        self.rnn2 = nn.LSTM(c_dim, c_dim, batch_first=True)
        self.rnn3 = nn.LSTM(c_dim, c_dim, batch_first=True)
        self.rnn4 = nn.LSTM(c_dim, c_dim, batch_first=True)

        self.english_rnn1 = nn.LSTM(512, 512, batch_first=True, bidirectional=True)
        self.english_rnn2 = nn.LSTM(1024, 1024, batch_first=True, bidirectional=True)
        self.relu = nn.ReLU()
        self.audio_encoder = nn.Sequential(
            nn.Linear(frame_dim // 2, frame_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(frame_dim // 4, 1),
            nn.LeakyReLU(),
        )

    def forward(self, mels):
        z = self.conv(mels)
        z = self.relu(z)
        z = self.encoder(z.transpose(1, 2))

        c, _ = self.rnn1(z)
        c, _ = self.rnn2(c)
        c, _ = self.rnn3(c)
        c, _ = self.rnn4(c)

        s, _ = self.english_rnn1(c)
        s, _ = self.english_rnn2(s)
        s = s.transpose(1, 2)
        s = self.audio_encoder(s)

        return z, z, s


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        model_weights = torch.load("pretrained/ckpt.pth")
        seed_model = alexnet(pretrained=False)
        self.image_model = nn.Sequential(*list(seed_model.features.children()))

        last_layer_index = len(list(self.image_model.children()))
        self.image_model.add_module(
            str(last_layer_index),
            nn.Conv2d(
                256,
                args["audio_model"]["embedding_dim"],
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
        )

        for name, param in self.image_model.named_parameters():
            print(name, param[0][0])
            break
        if args["pretrained_alexnet"]:
            print("Using pretrained AlexNet")
            model_dict = self.image_model.state_dict()
            for key in model_weights["model"]:
                if "features" in key:
                    new_key = ".".join(key.split(".")[2:])
                    model_dict[new_key] = model_weights["model"][key]
            self.image_model.load_state_dict(model_dict)

        for name, param in self.image_model.named_parameters():
            print(name, param[0][0])
            break

    def forward(self, x):
        x = self.image_model(x)
        x = x.view(x.size(0), x.size(1), -1)
        return x


class MattNet(nn.Module):
    def __init__(self, args):
        super(MattNet, self).__init__()
        self.audio_enc = AudioEncoder(args)
        self.image_enc = ImageEncoder(args)

    def forward(self, audio, image):
        audio_embedding = self.audio_enc(audio)
        image_embedding = self.image_enc(image)
        att = torch.bmm(audio_embedding.transpose(1, 2), image_embedding)
        sim, _ = att.max(dim=-1)
        sim, _ = sim.max(dim=-1)
        return sim.unsqueeze(-1)

    # def forward(self, embedding_1, embedding_2):
    #     att = torch.bmm(embedding_1.transpose(1, 2), embedding_2)
    #     s, _ = att.max(dim=-1)
    #     s, _ = s.max(dim=-1)
    #     return s.unsqueeze(-1)

    def score(self, embedding_1, embedding_2):
        scores = []
        for i in range(embedding_1.size(0)):

            att = torch.bmm(
                embedding_1[i, :, :].unsqueeze(0).transpose(1, 2),
                embedding_2[i, :, :].unsqueeze(0),
            )  # .squeeze(2)# / (torch.norm(aud_em, dim=1) * torch.norm(im, dim=1))
            s, _ = att.max(dim=-1)
            s, _ = s.max(dim=-1)
            scores.append(s.unsqueeze(-1))
        scores = torch.cat(scores, dim=0)
        return scores  # self.sig(scores)

    def attention_scores(self, image, audio):
        att = torch.bmm(audio.transpose(1, 2), image)
        ind = att.argmax().item()
        embedding = image[:, :, ind].squeeze()
        return embedding

    def get_attention(self, image, audio):
        return torch.bmm(audio.transpose(1, 2), image)

    def one_to_many_score(self, embeddings, embedding_1):
        scores = []
        for i in range(embeddings.size(0)):
            att = torch.bmm(
                embedding_1.transpose(1, 2), embeddings[i, :, :].unsqueeze(0)
            )
            s, _ = att.max(dim=-1)
            s, _ = s.max(dim=-1)
            scores.append(s.unsqueeze(-1))
        scores = torch.cat(scores, dim=1)
        return scores


MODELS = {
    "mattnet": MattNet,
}


def setup_model(*, model_name, **kwargs):
    return MODELS[model_name](**kwargs)
