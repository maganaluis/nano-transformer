# src/models.py

import math
import torch
import torch.nn as nn
import pytorch_lightning as pl

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Create constant positional encoding matrix with shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class DummyTransformer(pl.LightningModule):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=2048, dropout=0.1, input_vocab_size=1000, output_vocab_size=1000,
                 max_seq_length=128):
        super().__init__()
        self.save_hyperparameters()

        # Embedding layers for source and target sequences
        self.src_tok_emb = nn.Embedding(input_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(output_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length)

        # Transformer module
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, dropout)
        # Final linear layer for output projection
        self.fc_out = nn.Linear(d_model, output_vocab_size)
    
    def forward(self, src, tgt):
        # src and tgt: (seq_len, batch_size)
        src_emb = self.src_tok_emb(src) * math.sqrt(self.hparams.d_model)
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.hparams.d_model)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)
        transformer_output = self.transformer(src_emb, tgt_emb)
        logits = self.fc_out(transformer_output)
        return logits
    
    def training_step(self, batch, batch_idx):
        src, tgt = batch
        # Use shifted target sequences for input and expected output
        tgt_input = tgt[:-1, :]
        tgt_expected = tgt[1:, :]
        logits = self(src, tgt_input)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt_expected.reshape(-1))
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
