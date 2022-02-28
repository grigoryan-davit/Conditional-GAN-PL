from os import listdir
from os.path import isfile, join
from typing import List


import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelPruning
from pytorch_lightning.loggers import MLFlowLogger
import torch
import yaml

from .model import ConditionalGan
from ..data.dataset_generator import CGANDataModule


def train_model(config: dict, train_dirs=List[str], val_dirs=List[str]) -> float:
    datamodule = CGANDataModule(train_dirs=train_dirs, val_dirs=val_dirs)

    callbacks = list()
    early_stopping_callback = EarlyStopping(
        monitor="validation_loss",
        patience=config["early_stopping"]["patience"],
        stopping_threshold=config["early_stopping"]["stopping_threshold"],
    )
    callbacks.append(early_stopping_callback)
    for pruning_amount in config["pruning_amount"]:
        for lr_D in config["lr_D"]:
            for lr_G in config["lr_G"]:
                torch.cuda.empty_cache()

                model = ConditionalGan(lr_D=lr_D, lr_G=lr_G)

                if pruning_amount is not None:
                    pruning_callback = ModelPruning(
                        "l1_unstructured",
                        amount=pruning_amount,
                        use_lottery_ticket_hypothesis=True,
                    )
                    callbacks.append(pruning_callback)

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
                    stochastic_weight_avg=config["training_tricks"][
                        "stochastic_weight_avg"
                    ],
                    precision=config["training_tricks"]["precision"],
                    auto_scale_batch_size=config["training_tricks"][
                        "autoscale_batchsize"
                    ],
                )
                trainer.fit(model, datamodule)
                torch.save(
                    model.state_dict(),
                    f"{config['savedir']}/lr_D:{lr_D}_lr_G:{lr_G}_pruning:{pruning_amount}.pt",
                )


if __name__ == "__main__":
    with open("../../config/cgan_config.yml", "r") as f:
        config = yaml.load(f)
    path_list = [
        join(config["data_dir"], f)
        for f in listdir(config["data_dir"])
        if isfile(join(config["data_dir"], f))
    ]
    train_model(config=config, data_dirs=path_list)
