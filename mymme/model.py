import torch

import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import alexnet


class AudioEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels=40,
        num_channels=64,
        kernel_size=4,
        stride=2,
        padding=1,
        z_dim=64,
        c_dim=512,
        frame_dim=256,
        use_pretrained_cpc=False,
    ):
        super(AudioEncoder, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            num_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )

        self.encoder = nn.Sequential(
            # 1
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # 2
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # 3
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # 4
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # 5
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, z_dim),
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

        if use_pretrained_cpc:

            def drop_prefix(s, prefix, sep="."):
                fst, *rest = s.split(sep)
                assert fst == prefix
                return ".".join(rest)

            path = "mymme/checkpoints/audio-model-cpc-epoch-1500.pt"
            model_weights = torch.load(path)
            model_weights = model_weights["acoustic_model"]
            model_weights = {
                drop_prefix(key, "module"): value
                for key, value in model_weights.items()
            }
            model_dict = self.state_dict()

            for key in model_weights:
                if key in model_dict:
                    model_dict[key] = model_weights[key]
                else:
                    print(f"WARN · Missing key: {key}")
            self.load_state_dict(model_dict)

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

        # return z, z, s
        return s


# class AudioEncoder(nn.Module):
#     def __init__(
#         self,
#         *,
#         in_channels,
#         channels=[128, 128, 128],
#         kernel_sizes=[5, 3, 3],
#         dilations=[1, 2, 2],
#         embedding_dim=2_048,
#     ):
#         super().__init__()
#         blocks = []
#         i = in_channels
#         for o, k, d in zip(channels, kernel_sizes, dilations):
#             blocks.append(self.make_block(i, o, k, d))
#             i = o
#
#         self.backbone = nn.Sequential(*blocks)
#         self.classifier = nn.Linear(
#             in_features=channels[-1],
#             out_features=embedding_dim,
#         )
#
#     def make_block(self, in_channels, out_channels, kernel_size, dilation):
#         return nn.Sequential(
#             nn.Conv1d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 dilation=dilation,
#             ),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(num_features=out_channels),
#         )
#
#     def average_pooling(self, x):
#         return x.mean(dim=-1)
#
#     def forward(self, audio):
#         # audio : B × D × T
#         x = self.backbone(audio)
#         x = self.average_pooling(x)
#         x = self.classifier(x)
#         return x.unsqueeze(-1)


class ImageEncoder(nn.Module):
    def __init__(self, *, embedding_dim=2048, use_pretrained_alexnet=True):
        super(ImageEncoder, self).__init__()
        seed_model = alexnet(pretrained=False)
        self.image_model = nn.Sequential(*list(seed_model.features.children()))

        last_layer_index = len(list(self.image_model.children()))
        self.image_model.add_module(
            str(last_layer_index),
            nn.Conv2d(
                256,
                embedding_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
        )

        if use_pretrained_alexnet:
            print("Using pretrained AlexNet")
            path = "mymme/checkpoints/alexnet-self-supervised.pth"
            model_weights = torch.load(path, map_location="cpu")
            model_dict = self.image_model.state_dict()
            for key in model_weights["model"]:
                if "features" in key:
                    new_key = ".".join(key.split(".")[2:])
                    model_dict[new_key] = model_weights["model"][key]
            self.image_model.load_state_dict(model_dict)

        # for name, param in self.image_model.named_parameters():
        #     if "features" in name:
        #         param.requires_grad = False

    def forward(self, x):
        x = self.image_model(x)
        x = x.view(x.size(0), x.size(1), -1)
        return x


class MattNet(nn.Module):
    def __init__(self, audio_encoder_kwargs, image_encoder_kwargs):
        super(MattNet, self).__init__()
        self.audio_enc = AudioEncoder(**audio_encoder_kwargs)
        self.image_enc = ImageEncoder(**image_encoder_kwargs)
        # self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def forward(self, audio, image):
        return self.score(audio, image, type="cross")

    def l2_normalize(self, x, dim):
        return x / x.norm(dim=dim, keepdim=True)

    def score(self, audio, image, type):
        # def compute1(a, i):
        #     att = torch.bmm(a.unsqueeze(0).transpose(1, 2), i.unsqueeze(0))
        #     s = att.max()
        #     return s

        EINSUM_OP = {
            "pair": "bda,bdi->bai",
            "cross": "xda,ydi->xyai",
        }
        op = EINSUM_OP[type]

        audio_embedding = self.audio_enc(audio)
        image_embedding = self.image_enc(image)

        # sim = torch.stack([torch.stack([compute1(a, i) for i in image_embedding]) for a in audio_embedding])
        # sim = torch.cat(att).to(audio.device)

        # audio_embedding = self.l2_normalize(audio_embedding, dim=1)
        # image_embedding = self.l2_normalize(image_embedding, dim=1)

        sim = torch.einsum(op, audio_embedding, image_embedding)
        sim, _ = sim.max(dim=-1)
        sim, _ = sim.max(dim=-1)

        # τ = torch.maximum(self.logit_scale.exp(), torch.tensor(100.0))
        τ = 1.0
        return τ * sim

    def compute_loss(self, audio, image, labels):
        """Input shapes:

        - audio:  B × (pos + neg) × D × T
        - image:  B × (pos + neg) × 3 × H × W
        - labels: B × (pos + neg)

        """

        def score_audio_image(audio_emb, image_emb):
            op = "bnda,bmdi->bnmai"
            sim = torch.einsum(op, audio_emb, image_emb)
            sim, _ = sim.max(dim=-1)
            sim, _ = sim.max(dim=-1)
            return sim

        B1, N1, D, T = audio.shape
        B2, N2, C, H, W = image.shape

        assert B1 == B2 and N1 == N2
        B = B1
        N = N1

        audio = audio.view(B * N, D, T)
        audio_emb = self.audio_enc(audio)
        audio_emb = audio_emb.view(B, N, *audio_emb.shape[1:])

        image = image.view(B * N, C, H, W)
        image_emb = self.image_enc(image)
        image_emb = image_emb.view(B, N, *image_emb.shape[1:])

        # assume one positive per batch and all positives are in first position
        assert labels.sum() == B
        assert labels[:, 0].sum() == B
        assert labels[:, 1:].sum() == 0
        # assert labels == torch.cat([torch.ones(B, 1), torch.zeros(B, N - 1)], dim=1)

        sim1 = score_audio_image(audio_emb[:, :1], image_emb)
        sim1 = sim1.squeeze(1)
        sim2 = score_audio_image(audio_emb, image_emb[:, :1])
        sim2 = sim2.squeeze(2)

        pred = torch.cat([sim1, sim2], dim=0)

        true = torch.zeros(2 * B).to(labels.device).long()
        return F.cross_entropy(pred, true)



MODELS = {
    "mattnet": MattNet,
}


def setup_model(*, model_name, **kwargs):
    return MODELS[model_name](**kwargs)
