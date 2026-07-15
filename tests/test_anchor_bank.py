import unittest

import torch

from vista.anchor_bank import AnchorBank


class AnchorBankTest(unittest.TestCase):
    def test_budget_and_unique_indices(self):
        bank = AnchorBank(torch.arange(1, 101), anchor_ratio=0.1)
        self.assertEqual(bank.budget, 10)
        self.assertEqual(bank.indices.numel(), 10)
        self.assertEqual(bank.indices.unique().numel(), 10)
        self.assertGreaterEqual(bank.indices.min(), 0)
        self.assertLess(bank.indices.max(), 100)

    def test_handles_flat_and_invalid_scores(self):
        bank = AnchorBank(
            torch.tensor([float("nan"), float("inf"), 0.0, 0.0]),
            anchor_ratio=0.5,
        )
        self.assertEqual(bank.indices.numel(), 2)
        self.assertTrue(torch.isfinite(bank.scores).all())

    def test_update_tracks_jaccard(self):
        bank = AnchorBank(
            torch.tensor([10.0, 9.0, 1.0, 1.0]),
            anchor_ratio=0.5,
            ema=0.0,
        )
        before = bank.indices.clone()
        jaccard = bank.update(torch.tensor([1.0, 1.0, 9.0, 10.0]))
        self.assertFalse(torch.equal(before, bank.indices))
        self.assertGreaterEqual(jaccard, 0.0)
        self.assertLessEqual(jaccard, 1.0)
        self.assertEqual(bank.metrics()["anchor_refresh_count"], 1)

    def test_rejects_wrong_update_shape(self):
        bank = AnchorBank(torch.ones(8), anchor_ratio=0.25)
        with self.assertRaisesRegex(ValueError, "does not match"):
            bank.update(torch.ones(7))


if __name__ == "__main__":
    unittest.main()
