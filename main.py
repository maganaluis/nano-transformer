# main.py

import hydra
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from nano_transformer.models import DummyTransformer
from nano_transformer.data import DummyDataset

import torch
import wandb
wandb.login()
torch.set_float32_matmul_precision('medium')

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Merge the full YAML string if provided
    if cfg.get("config_str"):
        extra_cfg = OmegaConf.create(cfg.config_str)
        cfg = OmegaConf.merge(cfg, extra_cfg)
    
    # Set up Weights & Biases logger using values from the config
    wandb_logger = WandbLogger(project=cfg.wandb.project)
    
    # Configure callbacks from configuration values
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        save_top_k=1,
        mode="min",
        filename="dummy_transformer-{epoch:02d}-{train_loss:.2f}"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        strategy=cfg.trainer.strategy
    )
    
    # Create the dataset and DataLoader
    dataset = DummyDataset(
        num_samples=cfg.dataset.num_samples,
        seq_length=cfg.dataset.seq_length,
        vocab_size=cfg.dataset.vocab_size
    )
    train_loader = DataLoader(dataset, batch_size=cfg.trainer.batch_size, shuffle=True)
    
    # Instantiate the model using configuration
    model = DummyTransformer(
        input_vocab_size=cfg.model.input_vocab_size,
        output_vocab_size=cfg.model.output_vocab_size,
        max_seq_length=cfg.model.max_seq_length
    )
    
    # Start training
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()