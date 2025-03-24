import torch
from nano_transformer.data import DummyDataset

def test_dummy_dataset():
    num_samples = 10
    seq_length = 30
    vocab_size = 1000
    
    dataset = DummyDataset(num_samples=num_samples, seq_length=seq_length, vocab_size=vocab_size)
    
    # Fetch one sample
    src, tgt = dataset[0]
    
    # Ensure the sequence length is as expected
    assert src.shape[0] == seq_length
    assert tgt.shape[0] == seq_length
    
    # Ensure token values are within the vocabulary range
    assert src.max() < vocab_size
    assert tgt.max() < vocab_size
