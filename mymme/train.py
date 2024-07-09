from pathlib import Path
from typing import Any

import torch

from torch import optim
from torch.nn import functional as F

import click

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, create_lr_scheduler_with_warmup
from ignite.handlers.early_stopping import EarlyStopping
from ignite.handlers.param_scheduler import CosineAnnealingScheduler
from ignite.metrics import Loss, RunningAverage
from ignite.utils import convert_tensor, manual_seed

from ignite.handlers.tensorboard_logger import (
    TensorboardLogger,
    global_step_from_engine,
)

from mymme.data import setup_data
from mymme.model import setup_model


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
}


def identity_loss(loss, _):
    return loss


# def info_nce_cross_entropy_loss(pred, true):
# def loss_dir(pred):
#     # pred1 = torch.concat((pred[0, :1], pred[0, true == 0]))
#     # assert true.sum() == 1
#     # pred1 = pred[0]
#     # pred1 = F.softmax(pred1, dim=0)
#     # loss = -pred1[0].log()
#     # return loss
#     pred1 = pred[true == 1]
#     pred1 = F.softmax(pred1, dim=1)
#     pred1 = pred1[:, true == 1].sum(dim=1)
#     loss = -torch.log(pred1).mean()
#     return loss

# loss1 = loss_dir(pred)
# loss2 = loss_dir(pred.T)
# return (loss1 + loss2) / 2

# def loss_dir(pred):
#     pred1 = pred[true == 1]
#     loss1 = F.mse_loss(pred1[:, true == 1], torch.tensor(100.0).to("cuda"))
#     loss2 = F.mse_loss(pred1[:, true == 0], torch.tensor(0.0).to("cuda"))
#     return (loss1 + loss2) / 2

# loss1 = loss_dir(pred)
# loss2 = loss_dir(pred.T)
# return (loss1 + loss2) / 2

# loss1 = F.cross_entropy(pred, true)
# loss2 = F.cross_entropy(pred.T, true)
# return (loss1 + loss2) / 2


def prepare_batch_fn(batch, device, non_blocking):
    batch = {k: convert_tensor(v, device, non_blocking) for k, v in batch.items()}
    inp = batch["audio"], batch["image"], batch["label"]
    out = batch["label"]
    return inp, out


def model_fn(model, inp):
    audio, image, labels = inp
    return model.compute_loss(audio, image, labels)


@click.command()
@click.argument("config_name", type=str)
def train(config_name: str):
    config = CONFIGS[config_name]
    manual_seed(config["seed"])

    # Setup output directory
    output_dir = Path(f"output/{config_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # config.output_dir = output_dir

    dataloader_train, dataloader_valid = setup_data(**config["data"])

    device = config["device"]
    model = setup_model(**config["model"])
    model.to(device=device)

    optimizer = optim.Adam(model.parameters(), **config["optimizer"])
    # optimizer = optim.Adam(
    #     [
    #         {"params": model.audio_enc.parameters(), "name": "audio-enc"},
    #         {"params": model.image_enc.parameters(), "name": "image-enc"},
    #     ],
    #     **config["optimizer"],
    # )

    metrics = {
        "loss": Loss(identity_loss, device=device),
    }

    # Trainer and evaluator
    trainer = create_supervised_trainer(
        model,
        optimizer,
        prepare_batch=prepare_batch_fn,
        model_fn=model_fn,
        loss_fn=identity_loss,
        device=device,
    )
    evaluator = create_supervised_evaluator(
        model,
        prepare_batch=prepare_batch_fn,
        model_fn=model_fn,
        device=device,
    )

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # Model checkpoint
    def score_func(engine):
        return -engine.state.metrics["loss"]

    model_dir = output_dir / "checkpoints"
    handler = ModelCheckpoint(
        model_dir,
        n_saved=config["n_saved"],
        create_dir=True,
        require_empty=True,
        score_name="neg-loss",
        score_function=score_func,
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED(every=1),
        handler,
        {"model": model},
    )

    # Early stopping
    # handler = EarlyStopping(config["patience"], score_func, trainer)
    # evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)

    metric = RunningAverage(output_transform=lambda x: x)
    metric.attach(trainer, "running-average-loss")

    # Print metrics to the stderr with `add_event_handler` API for training stats
    def print_metrics(engine, tag):
        if tag == "train":
            metrics_str = "loss: {:.3f} · loss avg: {:.3f} ◇ lr: {:f}".format(
                engine.state.output,
                engine.state.metrics["running-average-loss"],
                optimizer.param_groups[0]["lr"],
            )
        elif tag == "valid":
            metrics_str = "loss: {:.3f}".format(engine.state.metrics["loss"])
        else:
            assert False, "Unknown tag"
        print(
            "{:s} · {:4d} / {:4d} · {:s}".format(
                tag,
                engine.state.epoch,
                engine.state.iteration,
                metrics_str,
            )
        )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config["log_every_iters"]),
        print_metrics,
        tag="train",
    )

    num_steps_per_epoch = len(dataloader_train)
    warmup_duration = num_steps_per_epoch * config["warmup_epochs"]
    cycle_size = num_steps_per_epoch * (config["max_epochs"] - config["warmup_epochs"])
    base_scheduler = CosineAnnealingScheduler(
        optimizer,
        "lr",
        start_value=config["optimizer"]["lr"],
        end_value=1e-6,
        cycle_size=cycle_size,
    )
    scheduler = create_lr_scheduler_with_warmup(
        base_scheduler,
        warmup_start_value=1e-6,
        warmup_end_value=config["optimizer"]["lr"],
        warmup_duration=warmup_duration,
    )
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Run evaluation at every training epoch end with shortcut `on` decorator
    # API and print metrics to the stderr again with `add_event_handler` API for
    # evaluation stats.
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def _():
        evaluator.run(dataloader_valid)
        print_metrics(evaluator, "valid")

    # Create a logger
    log_dir = output_dir / "tb-logs"
    tb_logger = TensorboardLogger(log_dir=log_dir)

    # Attach the logger to the trainer to log training loss at each iteration
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=config["log_every_iters"]),
        tag="train",
        output_transform=lambda loss: {"loss": loss},
    )

    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="valid",
        metric_names=["loss"],
        global_step_transform=global_step_from_engine(trainer),
    )

    tb_logger.close()

    # Setup is done. Let's run the training.
    trainer.run(
        dataloader_train,
        max_epochs=config["max_epochs"],
        # epoch_length=config["epoch_length"],
    )


if __name__ == "__main__":
    train()
