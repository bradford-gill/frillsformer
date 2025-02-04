import unittest
import torch
from src.layers.mulithead_attention import MultiheadAttention

class TestMultiheadAttention(unittest.TestCase):
    def setUp(self):
        """Set up common parameters and instances for the tests."""
        self.input_dim = 512
        self.num_heads = 8
        self.dim_model = 512
        self.batch_size = 32
        self.seq_length = 10

        # Instantiate MultiheadAttention
        self.mha = MultiheadAttention(input_dim=self.input_dim, num_heads=self.num_heads, dim_model=self.dim_model)

    def test_forward_pass_no_mask(self):
        """Test the forward pass without a mask."""
        x = torch.rand(self.batch_size, self.seq_length, self.input_dim)  # Random input tensor
        output = self.mha(x, mask=None)

        # Assert output shape is correct
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.dim_model),
                         "Output shape mismatch for input without mask")

    def test_forward_pass_with_mask(self):
        """Test the forward pass with a mask."""
        x = torch.rand(self.batch_size, self.seq_length, self.input_dim)  # Random input tensor
        mask = torch.ones(self.batch_size, self.seq_length, self.seq_length)  # Example mask
        output = self.mha(x, mask=mask)

        # Assert output shape is correct
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.dim_model),
                         "Output shape mismatch for input with mask")

    def test_return_attention_weights(self):
        """Test the forward pass with return_attention=True."""
        x = torch.rand(self.batch_size, self.seq_length, self.input_dim)  # Random input tensor
        mask = torch.ones(self.batch_size, self.seq_length, self.seq_length)  # Example mask
        output, attn = self.mha(x, mask=mask, return_attention=True)

        # Assert output and attention shapes are correct
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.dim_model),
                         "Output shape mismatch when returning attention")
        self.assertEqual(attn.shape, (self.batch_size, self.num_heads, self.seq_length, self.seq_length),
                         "Attention shape mismatch")

    def test_invalid_input_dimensions(self):
        """Test the forward pass with invalid input dimensions."""
        with self.assertRaises(RuntimeError):
            invalid_x = torch.rand(self.batch_size, self.seq_length, self.input_dim - 1)  # Incorrect input dim
            self.mha(invalid_x)

    def test_mask_broadcasting(self):
        """Test the forward pass with a broadcasted mask."""
        x = torch.rand(self.batch_size, self.seq_length, self.input_dim)  # Random input tensor
        mask = torch.ones(self.seq_length, self.seq_length)  # Simpler mask
        output = self.mha(x, mask=mask)

        # Assert output shape is correct
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.dim_model),
                         "Output shape mismatch with broadcasted mask")

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the MultiheadAttention layer."""
        x = torch.rand(self.batch_size, self.seq_length, self.input_dim, requires_grad=True)  # Random input tensor
        mask = torch.ones(self.batch_size, self.seq_length, self.seq_length)  # Example mask

        # Forward pass
        output = self.mha(x, mask=mask)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertTrue((x.grad != 0).any())

if __name__ == "__main__":
    unittest.main()