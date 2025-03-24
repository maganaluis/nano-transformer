import torch
from nano_transformer.models import DummyTransformer

def test_forward_pass():
    seq_length = 30
    batch_size = 4
    vocab_size = 1000

    # Instantiate the model
    model = DummyTransformer(
        input_vocab_size=vocab_size,
        output_vocab_size=vocab_size,
        max_seq_length=seq_length
    )

    # Create dummy inputs with shape (seq_length, batch_size)
    src = torch.randint(0, vocab_size, (seq_length, batch_size))
    tgt = torch.randint(0, vocab_size, (seq_length, batch_size))
    
    # Get model output
    logits = model(src, tgt)
    
    # Check that the output shape matches expectations:
    # (seq_length, batch_size, vocab_size)
    assert logits.shape == (seq_length, batch_size, vocab_size)
