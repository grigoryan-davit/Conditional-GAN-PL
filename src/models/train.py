from typing import List

import mlflow
import optuna
from optuna.integration import PytorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelPruning
from pytorch_lightning.loggers import MLFlowLogger
import torch
import yaml

from .model import ConditionalGan
from ..data.dataset_generator import ColorizationDataModule


with open("../../config/cgan_config.yml", "r") as f:
    config = yaml.load(f)


def objective(
    trial: optuna.trial.Trial,
    config: dict,
    data_dir=List[str],
) -> float:
    """
    Hyperparam search with optuna.
    Only optimizing the learning rates and pruning amounts for now
    (and in a somewhat limited range),
    since the added cost is roughly exponential.
    However, this can be extended to any number of hyperparameters.
    """
    lr_D = trial.suggest_float("lr_D", 3e-4, 3e-5)
    lr_G = trial.suggest_float("lr_G", 3e-4, 3e-5)
    pruning_rate = trial.suggest_float("pruning_rate", 0.1, 0.3)
    model = ConditionalGan(lr_D=lr_D, lr_G=lr_G)
    datamodule = ColorizationDataModule(data_dir=data_dir)  ## add datadir

    callbacks = list()
    if config["pruning_amount"] is not None:
        pruning_callback = ModelPruning(
            "l1_unstructured",
            amount=pruning_rate,
            use_lottery_ticket_hypothesis=True,
        )
        callbacks.append(pruning_callback)

    early_stopping_callback = EarlyStopping(
        monitor="validation_loss",
        patience=config["early_stopping"]["patience"],
        stopping_threshold=config["early_stopping"]["stopping_threshold"],
    )
    callbacks.append(early_stopping_callback)

    mlflow.pytorch.autolog()
    logger = MLFlowLogger(
        experiment_name=config["experiment_name"],
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=config["num_epochs"],
        progress_bar_refresh_rate=1,
        log_every_n_steps=100,
        check_val_every_n_epoch=1,
        gradient_clip_val=config["training_tricks"]["gradient_clip"],
        stochastic_weight_avg=config["training_tricks"]["stochastic_weight_avg"],
        precision=config["training_tricks"]["precision"],
        auto_scale_batch_size=config["training_tricks"]["autoscale_batchsize"],
    )

    with mlflow.start_run():
        trainer.fit(model, datamodule)
    return trainer.logged_metrics  #### change this


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    torch.save(
        model.state_dict(),
        f"{config['savedir']}/lookahead_all_texts_{lr}_{seed}_{rate}.pt",
    )
