# tests/test_model.py
import torch
import unittest
from src.model import Transformer

class TestTransformer(unittest.TestCase):
    def setUp(self):
        # Common setup for all tests
        self.batch_size = 2
        self.src_seq_len = 10
        self.tgt_seq_len = 12
        self.src_vocab_size = 100
        self.tgt_vocab_size = 150
        self.embed_dim = 512

        self.model = Transformer(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            embed_dim=self.embed_dim,
            max_seq_len=20,
            num_layers=2,
            num_heads=4,
            ff_dim=2048,
        )

    def test_forward_pass(self):
        # Dummy input (batch_size, seq_len)
        src = torch.randint(0, self.src_vocab_size, (self.batch_size, self.src_seq_len))
        tgt = torch.randint(0, self.tgt_vocab_size, (self.batch_size, self.tgt_seq_len))

        # Forward pass
        logits = self.model(src, tgt)

        # Check output shape
        self.assertEqual(logits.shape, (self.batch_size, self.tgt_seq_len, self.tgt_vocab_size))

    # def test_masking(self):
    #     # Test padding mask and look-ahead mask
    #     src = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    #     tgt = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])

    #     # Generate masks
    #     src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    #     tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1)), diagonal=1).bool()

    #     # Forward pass with masks
    #     logits = self.model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

    #     # Ensure no NaNs in output
    #     self.assertFalse(torch.isnan(logits).any(), "Output contains NaNs (masking failed?)")

if __name__ == "__main__":
    unittest.main()