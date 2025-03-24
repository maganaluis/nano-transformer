# main.py

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from nano_transformer.models import DummyTransformer
from nano_transformer.data import DummyDataset

import torch

torch.set_float32_matmul_precision('medium')

def main():
    # Set up Weights & Biases logger
    wandb_logger = WandbLogger(project="dummy_transformer_project", name="dummy_transformer_run")
    
    # Configure callbacks:
    # ModelCheckpoint to save the best model based on training loss.
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        save_top_k=1,
        mode="min",
        filename="dummy_transformer-{epoch:02d}-{train_loss:.2f}"
    )
    # LearningRateMonitor to log the learning rate at every step.
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Instantiate the PyTorch Lightning Trainer with the WandB logger and callbacks.
    trainer = pl.Trainer(
        max_epochs=10,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10
    )
    
    # Set up the dataset and dataloader.
    batch_size = 32
    seq_length = 30
    vocab_size = 1000
    dataset = DummyDataset(num_samples=1000, seq_length=seq_length, vocab_size=vocab_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Instantiate the transformer model.
    model = DummyTransformer(
        input_vocab_size=vocab_size,
        output_vocab_size=vocab_size,
        max_seq_length=seq_length
    )
    
    # Start training.
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()
