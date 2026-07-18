"""
The evaluator model: a shared transformer encoder with three task heads.

The evaluator (Objective C) must do more than binary detection, because the CHI
paper (Lagos Rojas et al., 2026) shows that a binary "is this a microaggression?"
LLM judge ceiling-rates everything with near-zero variance. Discrimination comes
from predicting *how severe* and *which kind*, not just *whether*. Hence three
heads:

  binary    -> misogynistic? (present for every training example)
  severity  -> 0-1000 intensity      (Biasly only)
  category  -> 6-way multi-label      (Biasly + partial Guest)

Partial labels are the defining constraint (evaluator.md §2): most
rows have no severity and no category. The loss for a head is therefore MASKED to
the rows that actually carry that label, so an absent label contributes nothing
rather than being treated as a zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel

from synthetic_conversation_generation.data_models.microaggression_taxonomy import CATEGORY_KEYS

N_CATEGORIES = len(CATEGORY_KEYS)
SEVERITY_MAX = 1000.0   # Biasly's scale; severity is regressed in [0, 1] then rescaled.


@dataclass
class EvaluatorOutput:
    loss: torch.Tensor | None
    binary_logit: torch.Tensor      # (B,)
    severity: torch.Tensor          # (B,) in [0, 1000]
    category_logits: torch.Tensor   # (B, N_CATEGORIES)


class MultiHeadEvaluator(nn.Module):
    """
    Shared encoder + {binary, severity, category} heads.

    Args:
        model_name:    HF encoder id (default RoBERTa-base — stable; DeBERTa-v3 is unstable to fine-tune).
        binary_pos_weight: weight for the positive class in the binary loss, to
                       counter the 19% positive rate. Passed from the data (n_neg/n_pos).
        lambda_severity / lambda_category: relative weights of the auxiliary losses.
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        binary_pos_weight: float = 4.0,
        lambda_severity: float = 1.0,
        lambda_category: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(hidden, 1)
        self.severity_head = nn.Linear(hidden, 1)
        self.category_head = nn.Linear(hidden, N_CATEGORIES)

        self.lambda_severity = lambda_severity
        self.lambda_category = lambda_category

        # Reduction is "none" so we can apply per-row masks before averaging.
        self.register_buffer("_pos_weight", torch.tensor(binary_pos_weight))
        self._bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=self._pos_weight)
        self._bce_cat = nn.BCEWithLogitsLoss(reduction="none")
        self._mse = nn.MSELoss(reduction="none")

    def _pool(self, input_ids, attention_mask) -> torch.Tensor:
        """Mean-pool the last hidden state over real (non-pad) tokens.

        Mean pooling rather than the [CLS] token: DeBERTa has no next-sentence
        pretraining objective, so [CLS] is not a calibrated sequence summary, and
        masked mean pooling is the robust default across encoder families.
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                      # (B, T, H)
        mask = attention_mask.unsqueeze(-1).float()         # (B, T, 1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(
        self,
        input_ids,
        attention_mask,
        binary_label=None,           # (B,) float 0/1
        severity_label=None,         # (B,) float 0-1000, NaN where absent
        category_label=None,         # (B, N) float multi-hot
        category_mask=None,          # (B,) bool — True where the row has category labels
    ) -> EvaluatorOutput:
        pooled = self.dropout(self._pool(input_ids, attention_mask))

        binary_logit = self.binary_head(pooled).squeeze(-1)
        severity = torch.sigmoid(self.severity_head(pooled).squeeze(-1)) * SEVERITY_MAX
        category_logits = self.category_head(pooled)

        loss = None
        if binary_label is not None:
            # Binary loss — every row contributes.
            loss = self._bce(binary_logit, binary_label).mean()

            # Severity loss — only rows with a (non-NaN) severity label.
            if severity_label is not None:
                sev_mask = ~torch.isnan(severity_label)
                if sev_mask.any():
                    pred = severity[sev_mask] / SEVERITY_MAX
                    target = severity_label[sev_mask] / SEVERITY_MAX
                    loss = loss + self.lambda_severity * self._mse(pred, target).mean()

            # Category loss — only rows flagged as carrying category labels.
            if category_label is not None and category_mask is not None and category_mask.any():
                cl = self._bce_cat(category_logits[category_mask], category_label[category_mask])
                loss = loss + self.lambda_category * cl.mean()

        return EvaluatorOutput(
            loss=loss,
            binary_logit=binary_logit,
            severity=severity,
            category_logits=category_logits,
        )
