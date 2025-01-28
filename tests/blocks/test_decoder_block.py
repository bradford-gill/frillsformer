import unittest
import torch
from src.blocks.decoder_block import DecoderBlock 

class TestDecoderBlock(unittest.TestCase):

    def test_decoder_block_output_shape(self):
        embed_dim = 512
        num_heads = 8
        ff_dim = 2048
        seq_length = 20
        batch_size = 16

        decoder_block = DecoderBlock(embed_dim, num_heads, ff_dim)

        # Mock inputs
        x = torch.rand(batch_size, seq_length, embed_dim)  # Decoder input
        encoder_output = torch.rand(batch_size, seq_length, embed_dim)  # Encoder output
        tgt_mask = torch.ones(batch_size, seq_length, seq_length)  # Target mask
        memory_mask = torch.ones(batch_size, seq_length, seq_length)  # Encoder-decoder mask

        # Forward pass
        output = decoder_block(x, encoder_output, tgt_mask, memory_mask)

        # Check that the output shape matches the input shape
        self.assertEqual(output.shape, x.shape)

    def test_decoder_block_no_mask(self):
        embed_dim = 128
        num_heads = 4
        ff_dim = 512
        seq_length = 10
        batch_size = 8

        decoder_block = DecoderBlock(embed_dim, num_heads, ff_dim)

        # Mock inputs without masks
        x = torch.rand(batch_size, seq_length, embed_dim)  # Decoder input
        encoder_output = torch.rand(batch_size, seq_length, embed_dim)  # Encoder output

        # Forward pass without masks
        output = decoder_block(x, encoder_output)

        # Check that the output shape matches the input shape
        self.assertEqual(output.shape, x.shape)

    def test_decoder_block_gradients(self):
        embed_dim = 64
        num_heads = 2
        ff_dim = 256
        seq_length = 5
        batch_size = 4

        decoder_block = DecoderBlock(embed_dim, num_heads, ff_dim)

        # Mock inputs with gradients enabled
        x = torch.rand(batch_size, seq_length, embed_dim, requires_grad=True)
        encoder_output = torch.rand(batch_size, seq_length, embed_dim, requires_grad=True)
        tgt_mask = torch.ones(batch_size, seq_length, seq_length)

        # Forward pass
        output = decoder_block(x, encoder_output, tgt_mask)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertTrue((x.grad != 0).any())

if __name__ == "__main__":
    unittest.main()
