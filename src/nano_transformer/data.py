# src/data.py

import torch
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=30, vocab_size=1000):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random token sequences for both source and target
        src = torch.randint(0, self.vocab_size, (self.seq_length,))
        tgt = torch.randint(0, self.vocab_size, (self.seq_length,))
        return src, tgt
