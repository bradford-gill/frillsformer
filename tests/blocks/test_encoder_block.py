import torch
import unittest
from src.blocks.encoder_block import EncoderBlock

class TestEncoderBlock(unittest.TestCase):
    def test_encoder_block_shapes(self):
        embed_dim = 64
        num_heads = 8
        ff_dim = 256
        batch_size = 4
        seq_length = 10

        encoder_block = EncoderBlock(embed_dim, num_heads, ff_dim)
        input_tensor = torch.rand(batch_size, seq_length, embed_dim)
        mask = torch.ones(batch_size, seq_length, seq_length)

        output_tensor = encoder_block(input_tensor, mask)
        self.assertEqual(output_tensor.shape, input_tensor.shape)

    def test_encoder_block_edge_cases(self):
        embed_dim = 64
        num_heads = 8
        ff_dim = 256

        # Single token
        encoder_block = EncoderBlock(embed_dim, num_heads, ff_dim)
        single_token_tensor = torch.rand(1, 1, embed_dim)
        mask = torch.ones(1, 1, 1)
        output_tensor = encoder_block(single_token_tensor, mask)
        self.assertEqual(output_tensor.shape, single_token_tensor.shape)

        # No tokens
        no_token_tensor = torch.rand(1, 0, embed_dim)
        mask = torch.ones(1, 0, 0)
        output_tensor = encoder_block(no_token_tensor, mask)
        self.assertEqual(output_tensor.shape, no_token_tensor.shape)

    def test_encoder_block_zero_output(self):
        embed_dim = 64
        num_heads = 8
        ff_dim = 256
        batch_size = 4
        seq_length = 10

        encoder_block = EncoderBlock(embed_dim, num_heads, ff_dim)
        input_tensor = torch.rand(batch_size, seq_length, embed_dim)
        mask = torch.ones(batch_size, seq_length, seq_length)
        zero_tensor = torch.zeros(batch_size, seq_length, embed_dim)

        output_tensor = encoder_block(input_tensor, mask)

        # Check that output is normalized correctly when input is non-zero
        self.assertNotEqual(torch.sum(output_tensor), 0)

    def test_encoder_block_gradients(self):
        embed_dim = 64
        num_heads = 8
        ff_dim = 256
        batch_size = 4
        seq_length = 10

        encoder_block = EncoderBlock(embed_dim, num_heads, ff_dim)
        input_tensor = torch.rand(batch_size, seq_length, embed_dim, requires_grad=True)
        mask = torch.ones(batch_size, seq_length, seq_length)

        # Forward pass
        output_tensor = encoder_block(input_tensor, mask)

        # Backward pass
        loss = output_tensor.sum()
        loss.backward()

        # Check gradients
        self.assertIsNotNone(input_tensor.grad)
        self.assertFalse(torch.isnan(input_tensor.grad).any())
        self.assertTrue((input_tensor.grad != 0).any())

if __name__ == "__main__":
    unittest.main()