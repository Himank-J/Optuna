import os
from pathlib import Path
import logging

import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import List

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Imports that require root directory setup
from src.utils.logging_utils import setup_logger, task_wrapper

log = logging.getLogger(__name__)


def instantiate_callbacks(callback_cfg: DictConfig, checkpoint_dir: Path) -> List[L.Callback]:
    callbacks: List[L.Callback] = []
    if not callback_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    for cb_name, cb_conf in callback_cfg.items():
        if "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            if cb_name == "model_checkpoint":
                # Modify the ModelCheckpoint callback to save in the checkpoint directory
                cb_conf.dirpath = str(checkpoint_dir)
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    loggers: List[Logger] = []
    if not logger_cfg:
        log.warning("No logger configs found! Skipping..")
        return loggers

    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers


@task_wrapper
def train(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
):
    log.info("Starting training!")
    trainer.fit(model, datamodule)
    train_metrics = trainer.callback_metrics
    log.info(f"Training metrics:\n{train_metrics}")
    return train_metrics


@task_wrapper
def test(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
):
    log.info("Starting testing!")
    if trainer.checkpoint_callback.best_model_path:
        log.info(
            f"Loading best checkpoint: {trainer.checkpoint_callback.best_model_path}"
        )
        test_metrics = trainer.test(
            model, datamodule, ckpt_path=trainer.checkpoint_callback.best_model_path
        )
    else:
        log.warning("No checkpoint found! Using current model weights.")
        test_metrics = trainer.test(model, datamodule)
    log.info(f"Test metrics:\n{test_metrics}")
    return test_metrics[0] if test_metrics else {}


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Set up paths
    log_dir = Path(cfg.paths.log_dir)
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)

    # Set up logger
    setup_logger(log_dir / "train_log.log")

    # Initialize DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Initialize Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    # Set up callbacks
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"), checkpoint_dir)

    # Set up loggers
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Ensure checkpoint directory exists
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Initialize Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=str(checkpoint_dir),
    )

    # Initialize empty dictionaries for metrics
    train_metrics = {}
    test_metrics = {}

    # Train the model
    if cfg.get("train"):
        train_metrics = train(cfg, trainer, model, datamodule)

    # Test the model
    if cfg.get("test"):
        test_metrics = test(cfg, trainer, model, datamodule)

    # Combine train and test metrics
    all_metrics = {**train_metrics, **test_metrics}

    # Extract the optimized metric
    optimized_metric = all_metrics.get(cfg.get("optimized_metric"))

    if optimized_metric is not None:
        log.info(f"Optimized metric value: {optimized_metric}")
    else:
        log.warning(f"Optimized metric '{cfg.get('optimized_metric')}' not found in the metrics.")

    # Return the optimized metric for Optuna
    return optimized_metric


if __name__ == "__main__":
    main()
