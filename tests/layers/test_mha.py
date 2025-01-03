import torch
import pytest
from src.layers.mulithead_attention import MultiheadAttention  # Replace with the actual path

def test_multihead_attention():
    # Parameters for testing
    input_dim = 512
    num_heads = 8
    dim_model = 512
    batch_size = 32
    seq_length = 10

    # Instantiate MultiheadAttention
    mha = MultiheadAttention(input_dim=input_dim, num_heads=num_heads, dim_model=dim_model)

    # Test case 1: Forward pass without a mask
    x = torch.rand(batch_size, seq_length, input_dim)  # Random input tensor
    output = mha(x, mask=None)

    # Assert output shape is correct
    assert output.shape == (batch_size, seq_length, dim_model), "Output shape mismatch for input without mask"

    # Test case 2: Forward pass with a mask
    mask = torch.ones(batch_size, seq_length, seq_length)  # Example mask
    output_with_mask = mha(x, mask=mask)

    # Assert output shape is correct
    assert output_with_mask.shape == (batch_size, seq_length, dim_model), "Output shape mismatch for input with mask"

    # Test case 3: Return attention weights
    output, attn = mha(x, mask=mask, return_attention=True)

    # Assert output and attention shapes are correct
    assert output.shape == (batch_size, seq_length, dim_model), "Output shape mismatch when returning attention"
    assert attn.shape == (batch_size, num_heads, seq_length, seq_length), "Attention shape mismatch"

    # Test case 4: Invalid input dimensions
    with pytest.raises(RuntimeError):
        invalid_x = torch.rand(batch_size, seq_length, input_dim - 1)  # Incorrect input dim
        mha(invalid_x)

    # Test case 5: Mask broadcasting
    mask = torch.ones(seq_length, seq_length)  # Simpler mask
    output = mha(x, mask=mask)
    assert output.shape == (batch_size, seq_length, dim_model), "Output shape mismatch with broadcasted mask"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
