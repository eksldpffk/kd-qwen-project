from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F



def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom



def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute:
        total_loss = alpha * CE + (1 - alpha) * KL(P_T || P_S)

    CE and KL are both computed only on valid response tokens.
    Because this is a causal LM, we shift logits/labels for next-token prediction.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    shift_student = student_logits[:, :-1, :].contiguous()
    shift_teacher = teacher_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    vocab_size = shift_student.size(-1)

    ce_loss = F.cross_entropy(
        shift_student.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    valid_mask = shift_labels.ne(-100)

    teacher_probs = F.softmax(shift_teacher, dim=-1)
    student_log_probs = F.log_softmax(shift_student, dim=-1)

    # KL(P_T || P_S) = sum_i p_i^T * (log p_i^T - log p_i^S)
    per_token_kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
        log_target=False,
    ).sum(dim=-1)

    kd_loss = masked_mean(per_token_kl, valid_mask)
    total_loss = alpha * ce_loss + (1.0 - alpha) * kd_loss

    return total_loss, ce_loss.detach(), kd_loss.detach()
