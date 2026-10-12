import unittest

import torch
from transformers import EsmConfig

from grade.ifm import GradeModel


class TargetCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        config = EsmConfig(
            vocab_size=33,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=32,
            max_position_embeddings=32,
            pad_token_id=1,
            mask_token_id=32,
            token_dropout=False,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        self.model = GradeModel(
            config,
            num_cross_attention_blocks=1,
            hidden_size=16,
            attn_implementation="eager",
        ).eval()

    def test_cached_target_scores_match_reference_path(self) -> None:
        target_ids = torch.tensor([[0, 5, 6, 7, 8, 2, 1, 1]])
        target_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
        peptide_ids = torch.tensor(
            [
                [0, 9, 10, 11, 2, 1, 1, 1],
                [0, 12, 13, 14, 15, 2, 1, 1],
                [0, 16, 17, 2, 1, 1, 1, 1],
            ]
        )
        peptide_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
            ]
        )
        batch_size = peptide_ids.shape[0]
        input_ids = torch.stack(
            [target_ids.expand(batch_size, -1), peptide_ids], dim=1
        )
        attention_mask = torch.stack(
            [target_mask.expand(batch_size, -1), peptide_mask], dim=1
        )

        with torch.no_grad():
            reference_logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )[0]
            target_encoding = self.model.encode_target(
                input_ids=target_ids,
                attention_mask=target_mask,
            )
            cached_logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_encoding=target_encoding,
            )[0]

        torch.testing.assert_close(
            cached_logits,
            reference_logits,
            rtol=1e-5,
            atol=1e-6,
        )
        reference_ranking = torch.argsort(
            torch.softmax(reference_logits, dim=-1)[:, 1],
            descending=True,
        )
        cached_ranking = torch.argsort(
            torch.softmax(cached_logits, dim=-1)[:, 1],
            descending=True,
        )
        self.assertTrue(torch.equal(cached_ranking, reference_ranking))

    def test_cached_target_batch_dimension_must_be_compatible(self) -> None:
        input_ids = torch.ones((3, 2, 8), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        incompatible_encoding = torch.zeros((2, 8, 16))

        with self.assertRaisesRegex(ValueError, "batch dimension"):
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_encoding=incompatible_encoding,
            )


if __name__ == "__main__":
    unittest.main()
