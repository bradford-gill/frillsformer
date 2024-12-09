import unittest
import torch
from src.layers import FeedForward


class TestFeedForward(unittest.TestCase):

    def test_feed_forward_shapes(self):
        d_model = 512
        d_ffn = 2048
        batch_size = 32
        seq_length = 50

        feed_forward = FeedForward(d_model, d_ffn)
        input_tensor = torch.rand(batch_size, seq_length, d_model)
        output_tensor = feed_forward(input_tensor)

        self.assertEqual(output_tensor.shape, input_tensor.shape)

    def test_feed_forward_edge_cases(self):
        d_model = 128
        d_ffn = 512

        # Single token
        feed_forward = FeedForward(d_model, d_ffn)
        single_token_tensor = torch.rand(1, 1, d_model)
        output_tensor = feed_forward(single_token_tensor)
        assert output_tensor.shape == single_token_tensor.shape

        # No tokens
        no_token_tensor = torch.rand(1, 0, d_model)
        output_tensor = feed_forward(no_token_tensor)
        assert output_tensor.shape == no_token_tensor.shape