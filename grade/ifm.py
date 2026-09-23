"""Information Fusion Module (IFM).

Fuses target (GPCR) and peptide representations via bidirectional
cross-attention. Uses two frozen protein sequence encoders — one per partner —
followed by parallel cross-attention blocks and a classifier head.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import EsmModel, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig


class CrossAttention(nn.Module):
    """Multi-head cross-attention layer (query ← key/value)."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query_embeddings, key_value_embeddings):
        attn_output, _ = self.cross_attention(query_embeddings, key_value_embeddings, key_value_embeddings)
        return attn_output


class GradeModel(PreTrainedModel):
    """Dual protein encoder + bidirectional cross-attention + classifier head.

    Both sequence encoders are frozen; only the cross-attention blocks and the
    classifier head are trainable.
    """

    config_class = PretrainedConfig
    base_model_prefix = "grade"

    def __init__(
        self,
        config: PretrainedConfig,
        num_cross_attention_blocks: int = 2,
        hidden_size: int = 640,
        *inputs,
        **kwargs,
    ):
        attn_implementation = kwargs.pop("attn_implementation", "sdpa")
        setattr(config, "_attn_implementation", attn_implementation)
        super().__init__(config, *inputs, **kwargs)
        self.gpcr_encoder = EsmModel(config)
        self.peptide_encoder = EsmModel(config)
        self.cross_attention_blocks = torch.nn.ModuleList([
            CrossAttention(embed_dim=hidden_size, num_heads=16) for _ in range(num_cross_attention_blocks)
        ])
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.classifier_head = torch.nn.Linear(hidden_size * 2 * num_cross_attention_blocks, 2)
        for param in self.gpcr_encoder.parameters():
            param.requires_grad = False
        for param in self.peptide_encoder.parameters():
            param.requires_grad = False
        self.post_init()

    def encode_target(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        """Return token-level GPCR embeddings for reuse across peptide batches."""
        return_dict = bool(
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        with torch.no_grad():
            outputs = self.gpcr_encoder(
                input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        return outputs[0]

    def _encode_peptide(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        output_attentions: Optional[bool],
        output_hidden_states: Optional[bool],
        return_dict: bool,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.peptide_encoder(
                input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        return outputs[0]

    def _classify_embeddings(
        self,
        gpcr_embeddings: torch.Tensor,
        peptide_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor],
    ):
        cross_attention_outputs = []
        for cross_attention_block in self.cross_attention_blocks:
            gpcr_attended_by_peptide = cross_attention_block(gpcr_embeddings, peptide_embeddings)
            peptide_attended_by_gpcr = cross_attention_block(peptide_embeddings, gpcr_embeddings)
            cross_attention_outputs.append(gpcr_attended_by_peptide)
            cross_attention_outputs.append(peptide_attended_by_gpcr)

        fused_embeddings = torch.cat(cross_attention_outputs, dim=-1)
        pooled_output = self.pooling(fused_embeddings.permute(0, 2, 1)).squeeze(-1)
        logits = self.classifier_head(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        output = (logits,)
        return ((loss,) + output) if loss is not None else output

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        target_encoding: Optional[torch.Tensor] = None,
    ):
        return_dict = bool(
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask are required")

        batch_size = input_ids.shape[0]
        if target_encoding is None:
            gpcr_embeddings = self.encode_target(
                input_ids[:, 0],
                attention_mask[:, 0],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            if target_encoding.ndim != 3:
                raise ValueError("target_encoding must have shape [batch, sequence, hidden]")
            if target_encoding.shape[0] == 1:
                gpcr_embeddings = target_encoding.expand(batch_size, -1, -1)
            elif target_encoding.shape[0] == batch_size:
                gpcr_embeddings = target_encoding
            else:
                raise ValueError(
                    "target_encoding batch dimension must be 1 or match input batch size"
                )

        peptide_embeddings = self._encode_peptide(
            input_ids[:, 1],
            attention_mask[:, 1],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return self._classify_embeddings(gpcr_embeddings, peptide_embeddings, labels)
