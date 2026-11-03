import unittest

import torch
from scipy.stats import rankdata, spearmanr

from rsr_core.loss import pair_loss, soft_rank_sigmoid, soft_spearman


class SoftRank(unittest.TestCase):
    def test_sharp_sigmoid_gives_hard_ranks(self):
        x = torch.tensor([0.3, -1.2, 2.5, 0.0, 1.1])
        soft = soft_rank_sigmoid(x, strength=1e4)
        self.assertTrue(torch.allclose(soft, torch.tensor(rankdata(x.numpy()), dtype=torch.float32), atol=1e-3))

    def test_sharp_soft_spearman_matches_scipy(self):
        g = torch.Generator().manual_seed(0)
        a, b = torch.randn(50, generator=g), torch.randn(50, generator=g)
        expected = spearmanr(a.numpy(), b.numpy())[0]
        got = soft_spearman(a, b, strength=1e4).item()
        # torch.std is the sample std and the numerator uses a population mean,
        # so the value is scaled by (n - 1) / n.
        self.assertAlmostEqual(got * 50 / 49, expected, places=3)

    def test_identical_and_reversed(self):
        x = torch.linspace(0, 1, 20)
        self.assertAlmostEqual(soft_spearman(x, x).item() * 20 / 19, 1.0, places=4)
        self.assertAlmostEqual(soft_spearman(x, -x).item() * 20 / 19, -1.0, places=4)


class PairLoss(unittest.TestCase):
    def setUp(self):
        self.words = ["a", "b", "c", "d"]
        self.vecs = torch.eye(4) + 0.1

    def test_too_few_scored_pairs_is_skipped(self):
        lookup = {("a", "b"): 0.5, ("b", "a"): 0.5}
        self.assertIsNone(pair_loss(self.vecs, self.words, lookup, min_pairs=2, strength=1.0))

    def test_only_scored_pairs_count(self):
        # cosines: a-b 0.995, c-d 0.707, a-c 0.0, which matches the score order.
        # b-c and b-d have no score and must not enter the loss.
        vecs = torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.7, 0.7]])
        lookup = {}
        for (x, y), s in {("a", "b"): 0.9, ("c", "d"): 0.5, ("a", "c"): 0.1}.items():
            lookup[(x, y)] = lookup[(y, x)] = s
        loss = pair_loss(vecs, self.words, lookup, min_pairs=3, strength=1e4)
        # perfect agreement on 3 pairs; rho carries the (n - 1) / n factor
        self.assertAlmostEqual(loss.item(), 1 - 2 / 3, places=3)

    def test_gradient_reaches_vectors(self):
        vecs = torch.randn(4, 8, requires_grad=True)
        lookup = {}
        for (x, y), s in {("a", "b"): 0.9, ("a", "c"): 0.1, ("b", "d"): 0.4, ("c", "d"): 0.5}.items():
            lookup[(x, y)] = lookup[(y, x)] = s
        pair_loss(vecs, self.words, lookup, min_pairs=3, strength=1.0).backward()
        self.assertGreater(vecs.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
