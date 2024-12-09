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