from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from policy import Policy


@dataclass
class ObjectiveOutput:
    loss: torch.Tensor       # skaliert auf pro-Token-Magnitude (fuer backward)
    loss_sum: float          # Summe der NLL ueber valide Tokens (fuer Logging)
    n_tokens: int            # Anzahl valider Tokens im Batch


class Objective(ABC):
    @abstractmethod
    def compute(self, batch: dict, policy: Policy) -> ObjectiveOutput:
        ...


class TeacherForcingObjective(Objective):
    """Standard-Cross-Entropy mit Pad-Ignore.

    Der zurueckgegebene `loss` ist pro-Token gemittelt (sum / n_tokens),
    damit die Gradientenmagnitude unabhaengig von Batch-Size und Padding ist.
    Fuer Logging wird zusaetzlich die unnormierte `loss_sum` mitgegeben,
    damit der Akkumulator korrekt gewichtet mitteln kann.
    """

    def __init__(self, pad_token: int):
        self.pad_token = int(pad_token)

    def compute(self, batch: dict, policy: Policy) -> ObjectiveOutput:
        device = next(policy.parameters()).device
        input_tokens = batch["input_tokens"].to(device, non_blocking=True)
        target_tokens = batch["target_tokens"].to(device, non_blocking=True)
        point_cloud = batch["point_cloud"].to(device, non_blocking=True)
        face_count = batch["face_count"].to(device, non_blocking=True)

        logits = policy.logits(input_tokens, point_cloud, face_count, None)

        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = target_tokens.reshape(-1)

        loss_sum_t = F.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=self.pad_token,
            reduction="sum",
        )
        n_tokens = int((flat_targets != self.pad_token).sum().item())

        if n_tokens == 0:
            zero = loss_sum_t * 0.0
            return ObjectiveOutput(loss=zero, loss_sum=0.0, n_tokens=0)

        loss = loss_sum_t / n_tokens
        return ObjectiveOutput(
            loss=loss,
            loss_sum=float(loss_sum_t.detach().item()),
            n_tokens=n_tokens,
        )
