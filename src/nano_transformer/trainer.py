# src/trainer.py

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from nano_transformer.models import DummyTransformer
from nano_transformer.data import DummyDataset

def train():
    # Hyperparameters for dummy data
    batch_size = 32
    seq_length = 30
    vocab_size = 1000

    # Create dataset and dataloader
    dataset = DummyDataset(num_samples=1000, seq_length=seq_length, vocab_size=vocab_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = DummyTransformer(input_vocab_size=vocab_size, output_vocab_size=vocab_size, max_seq_length=seq_length)
    
    # Set up the trainer and start training
    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=10)
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    train()
