from pathlib import Path
from typing import Any

import torch

from torch import optim
from torch.nn import functional as F

import click

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from ignite.handlers.early_stopping import EarlyStopping
from ignite.metrics import Loss
from ignite.utils import convert_tensor, manual_seed

from ignite.handlers.tensorboard_logger import (
    TensorboardLogger,
    global_step_from_engine,
)

from data import setup_data
from model import setup_model


CONFIGS = {
    "00": {
        "seed": 42,
        "device": "cuda",
        "max_epochs": 50,
        "epoch_length": 500,
        "n_saved": 3,
        "patience": 5,
        "log_every_iters": 10,
        "lr": 0.0001,
        "data": {
            "num_pos": 32,
            "num_neg": 256,
            # "num_pos": 1,
            # "num_neg": 12,
            "num_workers": 32,
        },
        "model": {
            "model_name": "mattnet",
            "audio_encoder_kwargs": {
                # Conv params
                "in_channels": 40,
                "embedding_dim": 2048,
                "num_channels": 64,
                "kernel_size": 4,
                "stride": 2,
                "padding": 1,
                # LSTM params
                "z_dim": 64,
                "c_dim": 512,
                # Last layer params
                "frame_dim": 256,
                "use_pretrained_cpc": False,
            },
            "image_encoder_kwargs": {
                "embedding_dim": 2048,
                "use_pretrained_alexnet": True,
            },
        },
    },
}


def info_nce_cross_entropy_loss(pred, true):
    def loss_dir(pred):
        pred1 = pred[true == 1]
        pred1 = F.softmax(pred1, dim=1)
        pred1 = pred1[:, true == 1].sum(dim=0)
        loss = -torch.log(pred1).mean()
        return loss

    loss1 = loss_dir(pred)
    loss2 = loss_dir(pred.T)
    return (loss1 + loss2) / 2

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

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    # loss_fn = nn.CrossEntropyLoss().to(device=device)

    metrics = {
        "loss": Loss(info_nce_cross_entropy_loss, device=device),
    }

    def prepare_batch_fn(batch, device, non_blocking):
        batch = {k: convert_tensor(v, device, non_blocking) for k, v in batch.items()}
        inp = batch["audio"], batch["image"]
        out = batch["label"]
        return inp, out

    def model_fn(model, inp):
        audio, image = inp
        return model(audio, image)

    # Trainer and evaluator
    trainer = create_supervised_trainer(
        model,
        optimizer,
        prepare_batch=prepare_batch_fn,
        model_fn=model_fn,
        loss_fn=info_nce_cross_entropy_loss,
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
    handler = EarlyStopping(config["patience"], score_func, trainer)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)

    # Print metrics to the stderr with `add_event_handler` API for training stats
    def print_metrics(engine, tag):
        if tag == "train":
            metrics_str = "loss: {:.3f}".format(engine.state.output)
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
        epoch_length=config["epoch_length"],
    )


if __name__ == "__main__":
    train()
