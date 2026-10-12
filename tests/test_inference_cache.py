import unittest

import torch
from transformers import EsmConfig

from grade.ifm import GradeModel
from grade.inference import score_batch


class InferenceTargetCacheTests(unittest.TestCase):
    def test_screening_run_encodes_target_once_across_batches(self) -> None:
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
        model = GradeModel(
            config,
            num_cross_attention_blocks=1,
            hidden_size=16,
            attn_implementation="eager",
        ).eval()
        target = torch.tensor([0, 5, 6, 7, 2, 1, 1, 1])
        target_mask = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0])

        def make_batch(peptides: torch.Tensor) -> dict[str, torch.Tensor]:
            batch_size = peptides.shape[0]
            peptide_masks = peptides.ne(1).long()
            return {
                "input_ids": torch.stack(
                    [target.expand(batch_size, -1), peptides], dim=1
                ),
                "attention_mask": torch.stack(
                    [target_mask.expand(batch_size, -1), peptide_masks], dim=1
                ),
            }

        batches = [
            make_batch(torch.tensor([[0, 8, 9, 2, 1, 1, 1, 1]])),
            make_batch(
                torch.tensor(
                    [
                        [0, 10, 11, 2, 1, 1, 1, 1],
                        [0, 12, 13, 14, 2, 1, 1, 1],
                    ]
                )
            ),
        ]
        encoder_calls = 0

        def count_encoder_call(_module, _args, _output) -> None:
            nonlocal encoder_calls
            encoder_calls += 1

        hook = model.gpcr_encoder.register_forward_hook(count_encoder_call)
        target_encoding = None
        try:
            with torch.no_grad():
                for batch in batches:
                    _, target_encoding = score_batch(
                        model,
                        batch,
                        target_encoding=target_encoding,
                        use_target_cache=True,
                    )
        finally:
            hook.remove()

        self.assertEqual(encoder_calls, 1)


if __name__ == "__main__":
    unittest.main()
