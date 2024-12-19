import unittest
import torch
from src.layers import ScaledDotProduct


class TestScaledDot: 
    def test_scaled_dot_product(self):
        # Set seed for reproducibility
        torch.manual_seed(0)

        # Initialize ScaledDotProduct module
        dim_k = 4
        dim_v = 6
        scaled_dot_product = ScaledDotProduct(dim_k)

        # Input tensors
        query = torch.rand(1, 2, dim_k)  # Batch size = 1, seq_len_q = 2, dim_k = 4
        key = torch.rand(1, 3, dim_k)    # Batch size = 1, seq_len_k = 3, dim_k = 4
        value = torch.rand(1, 3, dim_v)  # Batch size = 1, seq_len_k = 3, dim_v = 6
        mask = torch.tensor([[[1, 1, 0], [1, 1, 1]]])  # Mask of shape (batch_size, seq_len_q, seq_len_k)

        # Expected output shapes
        expected_output_shape = (1, 2, dim_v)  # Output shape: batch_size x seq_len_q x dim_v
        expected_attention_weights_shape = (1, 2, 3)  # Weights shape: batch_size x seq_len_q x seq_len_k

        # Run forward pass
        output, attention_weights = scaled_dot_product(query, key, value, mask)

        # Assertions for shapes
        assert output.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, got {output.shape}"
        assert attention_weights.shape == expected_attention_weights_shape, f"Expected weights shape {expected_attention_weights_shape}, got {attention_weights.shape}"

        # Check that attention weights sum to 1 along last dimension (excluding masked entries)
        valid_attention_weights = attention_weights * mask  # Apply mask
        assert torch.allclose(valid_attention_weights.sum(dim=-1), mask.sum(dim=-1).float()), \
            "Attention weights do not sum to valid masked positions"

        print("Test passed!")


if __name__ == "main":
    TestScaledDot().test_scaled_dot_product()