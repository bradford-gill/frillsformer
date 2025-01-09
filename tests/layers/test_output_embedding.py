import unittest
import torch
import torch.nn as nn
from src.layers import OutputEmbedding


class TestOutputEmbedding(unittest.TestCase):
    def test_output_embedding_shapes(self):
        vocab_size = 10000
        embed_dim = 512
        seq_len = 20
        batch_size = 32

        output_embedding = OutputEmbedding(vocab_size, embed_dim)
        decoder_outputs = torch.randn(batch_size, seq_len, embed_dim)
        logits = output_embedding(decoder_outputs)

        self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))

    def test_weight_tying(self):
        vocab_size = 5000
        embed_dim = 256

        input_embedding = nn.Embedding(vocab_size, embed_dim)
        output_embedding = OutputEmbedding(vocab_size, embed_dim, tie_weights=input_embedding)

        # Check if weights are tied
        self.assertTrue(torch.allclose(output_embedding.proj.weight, input_embedding.weight))

    def test_forward_pass(self):
        vocab_size = 8000
        embed_dim = 128
        seq_len = 10
        batch_size = 4

        output_embedding = OutputEmbedding(vocab_size, embed_dim)
        decoder_outputs = torch.randn(batch_size, seq_len, embed_dim)
        logits = output_embedding(decoder_outputs)

        # Check if output values are finite
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

if __name__ == "__main__":
    unittest.main()
