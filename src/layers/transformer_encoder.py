from src.layers.mulithead_attention import MultiheadAttention
from src.layers.add_norm import AddNorm
from src.layers.feed_forward import FeedForward


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(embed_dim, num_heads)
        self.add_norm_1 = AddNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.add_norm_2 = AddNorm(embed_dim)

    def forward(self, x, mask=None):
        # Self-attention + Add & Norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.add_norm_1(x, attn_output)

        # Feedforward + Add & Norm
        ff_output = self.feed_forward(x)
        x = self.add_norm_2(x, ff_output)
        return x
    

# DOwn here (or in a separate file) we then need to do a TransformerEncoder
# which stacks all the layers and applies them in a loop (just like how you had done in the call)
