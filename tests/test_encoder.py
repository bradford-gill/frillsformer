import unittest
import torch
from src.encoder import Encoder


class TestEncoder(unittest.TestCase):

    def test_encoder_output_shape(self):
        num_layers = 6
        embed_dim = 512
        num_heads = 8
        ff_dim = 2048
        dropout = 0.1
        seq_length = 50
        batch_size = 32

        x = torch.rand(batch_size, seq_length, embed_dim)
        mask = torch.ones(batch_size, seq_length, seq_length)

        encoder = Encoder(num_layers, embed_dim, num_heads, ff_dim, dropout)

        # Forward pass
        output = encoder(x, mask)

        # Check that the output shape matches the expected shape
        self.assertEqual(output.shape, x.shape)

    def test_encoder_no_mask(self):
        num_layers = 4
        embed_dim = 128
        num_heads = 4
        ff_dim = 512
        dropout = 0.1
        seq_length = 20
        batch_size = 16

        x = torch.rand(batch_size, seq_length, embed_dim)

        encoder = Encoder(num_layers, embed_dim, num_heads, ff_dim, dropout)

        # Forward pass without mask
        output = encoder(x)

        # Check that the output shape matches the expected shape
        self.assertEqual(output.shape, x.shape)

    def test_encoder_gradients(self):
        num_layers = 2
        embed_dim = 64
        num_heads = 2
        ff_dim = 256
        dropout = 0.1
        seq_length = 10
        batch_size = 8

        x = torch.rand(batch_size, seq_length, embed_dim, requires_grad=True)
        mask = torch.ones(batch_size, seq_length, seq_length)

        encoder = Encoder(num_layers, embed_dim, num_heads, ff_dim, dropout)

        output = encoder(x, mask)

        # Compute a simple loss and perform backward pass
        loss = output.sum()
        loss.backward()

        # Check that gradients are not None or NaN
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertTrue((x.grad != 0).any())

if __name__ == "__main__":
    unittest.main()
