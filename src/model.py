from layers import (AddNorm, FeedForward, MultiHeadAttention, InputEncoding, OutputEmbedding)
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim, embed_dim, n_heads, num_encoder_layers, num_decoder_layers, ff_dim, max_seq_length, num_classes):
        pass
    
    def forward(self, source, targer):
        pass