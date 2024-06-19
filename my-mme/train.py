from pathlib import Path
from typing import Any

from torch import nn, optim
from torch.utils.data import DataLoader

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from ignite.handlers.early_stopping import EarlyStopping
from ignite.metrics import Accuracy, Loss
from ignite.utils import convert_tensor, manual_seed

from ignite.handlers.tensorboard_logger import (
    TensorboardLogger,
    global_step_from_engine,
)

from speechbrain.dataio.batch import PaddedBatch

from data import setup_data
from model import setup_model


CONFIGS = {
    "baseline": {
        "seed": 42,
        "device": "cuda",
        "batch_size_train": 32,
        "batch_size_valid": 32,
        "num_workers": 0,
        "max_epochs": 30,
        "n_saved": 3,
        "patience": 5,
        "save_every_iters": 1000,
        "log_every_iters": 10,
        "lr": 0.0002,
        "features": {
            "n_mels": 24,
        },
        "model": {
            "model_name": "baseline",
            "in_channels": 24,
        },
    },
}


def train(config_name: str):
    config = CONFIGS[config_name]
    manual_seed(config["seed"])

    # Setup output directory
    output_dir = Path(f"output/{config_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # config.output_dir = output_dir

    dataloader_train, dataloader_eval = setup_data(config)

    device = config["device"]
    model = setup_model(**config["model"])
    model.to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss().to(device=device)

    metrics = {
        "accuracy": Accuracy(device=device),
        "loss": Loss(loss_fn, device=device),
    }

    def prepare_batch_fn(batch, device, non_blocking):
        x = batch.features
        y = batch.identity

        # Return a tuple of (x, y) that can be directly runned as
        # `loss_fn(model(x), y)`
        return (
            convert_tensor(x, device, non_blocking),
            convert_tensor(y, device, non_blocking),
        )

    # Trainer and evaluator
    trainer = create_supervised_trainer(
        model,
        optimizer,
        loss_fn,
        device=device,
        prepare_batch=prepare_batch_fn,
    )
    evaluator = create_supervised_evaluator(
        model,
        metrics=metrics,
        device=device,
        prepare_batch=prepare_batch_fn,
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
            metrics_str = "loss: {:.3f} · accuracy: {:.2f}%".format(
                engine.state.metrics["loss"],
                100 * engine.state.metrics["accuracy"],
            )
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
        evaluator.run(dataloader_eval)
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
        metric_names=["loss", "accuracy"],
        global_step_transform=global_step_from_engine(trainer),
    )

    tb_logger.close()

    # Setup is done. Let's run the training.
    trainer.run(
        dataloader_train,
        max_epochs=config["max_epochs"],
    )


if __name__ == "__main__":
    train("baseline")