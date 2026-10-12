"""Peptide Adaptive Module (PAM).

Responsible for peptide candidate library loading, format adaptation,
tokenization, and dataset/sampler construction for inference.
"""

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from .inputs import add_unique_peptide, load_peptides
from .tcm import normalize_sequence


@dataclass
class PeptideRecord:
    peptide_name: str
    peptide_seq: str


class GPCRPeptideDataset(Dataset):
    def __init__(self, target_name: str, target_seq: str, peptides: Dict[str, str]):
        self.target_name = target_name
        self.target_seq = target_seq
        self.records = [PeptideRecord(name, seq) for name, seq in peptides.items()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> PeptideRecord:
        return self.records[idx]


class SequentialDistributedSampler(Sampler[int]):
    """Distributed sampler that assigns contiguous shards without padding.

    Unlike ``DistributedSampler``, this does not pad the dataset to a multiple
    of ``world_size``, avoiding duplicate candidates in the output.
    """

    def __init__(self, dataset: Dataset, skip: int = 0):
        import torch.distributed as dist

        if not (int(os.environ.get("WORLD_SIZE", "1")) > 1):
            raise RuntimeError("SequentialDistributedSampler requires distributed initialization")
        self.dataset = dataset
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.skip = skip

    def __iter__(self):
        total_size = len(self.dataset)
        per_rank = math.ceil(total_size / self.world_size)
        start = self.rank * per_rank + self.skip
        end = min((self.rank + 1) * per_rank, total_size)
        if start >= end:
            return iter(())
        return iter(range(start, end))

    def __len__(self) -> int:
        total_size = len(self.dataset)
        per_rank = math.ceil(total_size / self.world_size)
        start = self.rank * per_rank + self.skip
        end = min((self.rank + 1) * per_rank, total_size)
        return max(0, end - start)


class SkipSampler(Sampler[int]):
    """Sequential sampler that skips the first N items — used for single-process resume."""

    def __init__(self, dataset: Dataset, skip: int):
        self.dataset = dataset
        self.skip = skip

    def __iter__(self):
        return iter(range(self.skip, len(self.dataset)))

    def __len__(self) -> int:
        return max(0, len(self.dataset) - self.skip)


class PairCollator:
    """Collate function that tokenizes (target, peptide) pairs for the model."""

    def __init__(self, tokenizer, target_seq: str, max_length: int):
        self.tokenizer = tokenizer
        self.target_seq = " ".join(list(normalize_sequence(target_seq)))
        self.max_length = max_length

    def __call__(self, batch: Sequence[PeptideRecord]) -> Dict[str, torch.Tensor]:
        seqs: List[str] = []
        peptide_names: List[str] = []
        peptide_seqs: List[str] = []
        for record in batch:
            peptide_seq = normalize_sequence(record.peptide_seq)
            seqs.append(self.target_seq)
            seqs.append(" ".join(list(peptide_seq)))
            peptide_names.append(record.peptide_name)
            peptide_seqs.append(peptide_seq)

        tokenized = self.tokenizer(
            seqs,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        features = {
            key: value.view(len(batch), 2, -1)
            for key, value in tokenized.items()
        }
        features["peptide_names"] = peptide_names
        features["peptide_seqs"] = peptide_seqs
        return features
