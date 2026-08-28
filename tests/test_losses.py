import math
import unittest

import torch

from losses import categorical_logps, dpo_loss, grpo_loss, sft_loss


class LossTests(unittest.TestCase):
    def test_sft_and_categorical_logps(self):
        logits = torch.zeros(2, 3, requires_grad=True)
        targets = torch.tensor([1, 2])
        loss = sft_loss(logits, targets)
        self.assertAlmostEqual(float(loss), math.log(3.0), places=6)
        gathered = categorical_logps(logits, targets)
        self.assertTrue(torch.allclose(gathered, torch.full((2,), -math.log(3.0))))

    def test_dpo_equal_policy_and_reference_is_log_two(self):
        chosen = torch.tensor([-2.0, -3.0])
        rejected = torch.tensor([-2.5, -4.0])
        output = dpo_loss(chosen, rejected, chosen, rejected, reduction="none")
        self.assertTrue(torch.allclose(output.loss, torch.full((2,), math.log(2.0))))

    def test_grpo_zero_variance_group_has_no_signal(self):
        logps = torch.tensor([-0.5, -0.6, -0.7], requires_grad=True)
        rewards = torch.tensor([1.0, 1.0, 1.0])
        groups = torch.tensor([0, 0, 0])
        output = grpo_loss(logps, rewards, groups)
        self.assertEqual(output.zero_variance_groups, 1)
        self.assertTrue(torch.allclose(output.advantages, torch.zeros(3)))
        output.loss.backward()
        self.assertTrue(torch.allclose(logps.grad, torch.zeros(3)))

    def test_grpo_advantage_signs_follow_group_rewards(self):
        logps = torch.tensor([-0.5, -0.6, -0.7], requires_grad=True)
        rewards = torch.tensor([0.0, 1.0, 2.0])
        groups = torch.tensor([4, 4, 4])
        output = grpo_loss(logps, rewards, groups)
        self.assertLess(float(output.advantages[0]), 0.0)
        self.assertAlmostEqual(float(output.advantages[1]), 0.0, places=5)
        self.assertGreater(float(output.advantages[2]), 0.0)


if __name__ == "__main__":
    unittest.main()
