import unittest
import torch
from src.layers import InputEncoding


class TestInputEncoding(unittest.TestCase):

    def test_input_encoding_shapes(self):
        vocab_size = 10000
        embed_dim = 512
        max_seq_len = 50
        batch_size = 32
        seq_length = 20

        input_encoding = InputEncoding(vocab_size, embed_dim, max_seq_len)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))  # Random token IDs
        output_tensor = input_encoding(input_ids)

        self.assertEqual(output_tensor.shape, (batch_size, seq_length, embed_dim))

    def test_input_encoding_edge_cases(self):
        vocab_size = 5000
        embed_dim = 256
        max_seq_len = 10

        # Single token
        input_encoding = InputEncoding(vocab_size, embed_dim, max_seq_len)
        single_token_input = torch.randint(0, vocab_size, (1, 1))  # Single sequence, single token
        output_tensor = input_encoding(single_token_input)
        self.assertEqual(output_tensor.shape, (1, 1, embed_dim))

        # No tokens
        no_token_input = torch.randint(0, vocab_size, (1, 0))  # Single sequence, no tokens
        output_tensor = input_encoding(no_token_input)
        self.assertEqual(output_tensor.shape, (1, 0, embed_dim))

    def test_input_encoding_positional_values(self):
        vocab_size = 10000
        embed_dim = 512
        max_seq_len = 50
        batch_size = 4
        seq_length = 10

        input_encoding = InputEncoding(vocab_size, embed_dim, max_seq_len)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        output_tensor = input_encoding(input_ids)

        # Test positional encoding contribution
        token_embeds = input_encoding.token_embedding(input_ids)
        positional_embeds = input_encoding.positional_encoding[:, :seq_length, :]
        self.assertTrue(torch.allclose(output_tensor, token_embeds + positional_embeds, atol=1e-6))

    def test_input_encoding_gradients(self):
        vocab_size = 10000
        embed_dim = 512
        max_seq_len = 50
        batch_size = 16
        seq_length = 20

        input_encoding = InputEncoding(vocab_size, embed_dim, max_seq_len)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        output_tensor = input_encoding(input_ids)

        # Compute a simple loss and backpropagate
        loss = output_tensor.sum()  # Simple scalar loss
        loss.backward()

        # Check gradients exist and are valid
        self.assertIsNotNone(input_encoding.token_embedding.weight.grad)
        self.assertFalse(torch.isnan(input_encoding.token_embedding.weight.grad).any())

    def test_input_encoding_position_independence(self):
        vocab_size = 10000
        embed_dim = 256
        max_seq_len = 50
        batch_size = 1
        seq_length = 10

        input_encoding = InputEncoding(vocab_size, embed_dim, max_seq_len)

        # Create identical input tokens at different positions
        input_ids = torch.tensor([[42] * seq_length])  # All tokens have the same value
        output_tensor = input_encoding(input_ids)

        # Check that outputs differ for different positions (due to positional encoding)
        self.assertFalse(torch.allclose(output_tensor[:, 0, :], output_tensor[:, 1, :], atol=1e-6))

if __name__ == "__main__":
    unittest.main()
