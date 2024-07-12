CONFIGS = {
    "00": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 50,
        # "epoch_length": 500,
        "n_saved": 3,
        "patience": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 0.0001,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 32,
            "num_neg": 256,
            "num_workers": 32,
            "num_word_repeats": 64,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "01": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 50,
        # "epoch_length": 500,
        "n_saved": 3,
        "patience": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 0.0001,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 32,
            "num_neg": 256,
            "num_workers": 32,
            "num_word_repeats": 64,
            "to_shuffle": False,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "02": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 5,
        "n_saved": 3,
        "patience": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 0.0001,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 8,
            "num_neg": 32,
            "num_workers": 32,
            "to_shuffle": False,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "03": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 5,
        "n_saved": 3,
        "patience": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 0.0001,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 9,
            "num_workers": 32,
            "batch_size": 12,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "04": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 12,
        "warmup_epochs": 2,
        "n_saved": 5,
        # "patience": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 3e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 32,
            "batch_size": 32,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "05": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 12,
        "warmup_epochs": 2,
        "n_saved": 5,
        # "patience": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 3e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 32,
            "batch_size": 32,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "06": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 12,
        "warmup_epochs": 2,
        "n_saved": 5,
        # "patience": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 3e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 32,
            "batch_size": 32,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
                "pooling_layer": "average",
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "07": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 12,
        "warmup_epochs": 2,
        "n_saved": 5,
        # "patience": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 3e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 32,
            "batch_size": 32,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
                "pooling_layer": "average",
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "08": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 24,
        "warmup_epochs": 2,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 3e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 32,
            "batch_size": 32,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
                "pooling_layer": "average",
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "09": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 24,
        "warmup_epochs": 2,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 2e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 32,
            "batch_size": 32,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
                "pooling_layer": "average",
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
    "10": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 32,
        "warmup_epochs": 2,
        "n_saved": 5,
        "log_every_iters": 5,
        "optimizer": {
            "lr": 2e-4,
            "weight_decay": 5e-7,
        },
        "data": {
            "langs": ("english",),
            "num_pos": 1,
            "num_neg": 11,
            "num_workers": 32,
            "batch_size": 48,
            "to_shuffle": True,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                "use_pretrained_cpc": False,
                "pooling_layer": "average",
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
}