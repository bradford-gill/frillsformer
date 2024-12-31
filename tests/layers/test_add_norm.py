import unittest
import torch
import torch.nn as nn
from src.layers import AddNorm


class TestAddNorm(unittest.TestCase):

    def test_add_norm_shapes(self):
        embed_dim = 512
        batch_size = 32
        seq_length = 50

        add_norm = AddNorm(embed_dim)
        input_tensor = torch.rand(batch_size, seq_length, embed_dim)
        sublayer_output = torch.rand(batch_size, seq_length, embed_dim)
        output_tensor = add_norm(input_tensor, sublayer_output)

        self.assertEqual(output_tensor.shape, input_tensor.shape)

    def test_add_norm_edge_cases(self):
        embed_dim = 128

        # Single token
        add_norm = AddNorm(embed_dim)
        single_token_tensor = torch.rand(1, 1, embed_dim)
        sublayer_output = torch.rand(1, 1, embed_dim)
        output_tensor = add_norm(single_token_tensor, sublayer_output)
        self.assertEqual(output_tensor.shape, single_token_tensor.shape)

        # No tokens
        no_token_tensor = torch.rand(1, 0, embed_dim)
        sublayer_output = torch.rand(1, 0, embed_dim)
        output_tensor = add_norm(no_token_tensor, sublayer_output)
        self.assertEqual(output_tensor.shape, no_token_tensor.shape)

    def test_add_norm_zero_output(self):
        # Ensuring AddNorm behaves correctly when sublayer_output is all zeros
        embed_dim = 256
        batch_size = 16
        seq_length = 20

        add_norm = AddNorm(embed_dim)
        input_tensor = torch.rand(batch_size, seq_length, embed_dim)
        sublayer_output = torch.zeros(batch_size, seq_length, embed_dim)
        output_tensor = add_norm(input_tensor, sublayer_output)

        # When sublayer_output is zero, the output should just be the normalized input
        # (Use the same LayerNorm instance from AddNorm)
        normalized_input = add_norm.norm(input_tensor)
        self.assertTrue(torch.allclose(output_tensor, normalized_input, atol=1e-6))

    def test_add_norm_gradients(self):
        embed_dim = 512
        batch_size = 16
        seq_length = 20

        add_norm = AddNorm(embed_dim)
        input_tensor = torch.rand(batch_size, seq_length, embed_dim, requires_grad=True)
        sublayer_output = torch.rand(batch_size, seq_length, embed_dim, requires_grad=True)

        # Forward pass
        output = add_norm(input_tensor, sublayer_output)

        # Backward pass
        loss = output.sum()  # Simple scalar loss
        loss.backward()

        # Check gradients
        self.assertIsNotNone(input_tensor.grad)
        self.assertIsNotNone(sublayer_output.grad)
        self.assertFalse(torch.isnan(input_tensor.grad).any())
        self.assertFalse(torch.isnan(sublayer_output.grad).any())



if __name__ == "__main__":
    unittest.main()
