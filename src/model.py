# model.py
import torch
import torch.nn as nn
from src.encoder import Encoder
from src.decoder import Decoder
from src.layers.input_encoding import InputEncoding
from src.layers.output_embedding import OutputEmbedding

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        # Encoder components
        self.encoder_embed = InputEncoding(src_vocab_size, embed_dim, max_seq_len)
        self.encoder = Encoder(num_layers, embed_dim, num_heads, ff_dim, dropout)
        
        # Decoder components
        self.decoder_embed = InputEncoding(tgt_vocab_size, embed_dim, max_seq_len)
        self.decoder = Decoder(num_layers, embed_dim, num_heads, ff_dim, dropout)
        
        # Output projection (shares weights with decoder embeddings)
        self.output = OutputEmbedding(
            tgt_vocab_size, 
            embed_dim, 
            tie_weights=self.decoder_embed.token_embedding
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Encoder forward
        src_emb = self.encoder_embed(src)
        encoder_out = self.encoder(src_emb, mask=src_mask)
        
        # Decoder forward
        tgt_emb = self.decoder_embed(tgt)
        decoder_out = self.decoder(
            x=tgt_emb,
            encoder_output=encoder_out,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        
        # Output logits
        logits = self.output(decoder_out)
        return logits