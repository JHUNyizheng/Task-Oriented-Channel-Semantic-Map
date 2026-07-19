from __future__ import annotations

import torch

from .heads import MultiTaskHeads, TASK_NAMES


class GatedHLG(torch.nn.Module):
    """Environment-aware hierarchical local-global operator.

    `gamma` is the neural-branch weight throughout the code and manuscript:
    output = gamma * neural + (1 - gamma) * local_prior.
    """

    def __init__(
        self,
        support_dim: int,
        query_dim: int,
        hidden: int,
        far_count: int,
        near_angle_count: int,
        near_range_count: int,
        attention_heads: int = 4,
        ablation: str | None = None,
        gate_evidence_features: bool = False,
    ) -> None:
        super().__init__()
        self.environment_dim = query_dim - 4
        if self.environment_dim < 1:
            raise ValueError("query tokens must contain xyz, environment features, and support ratio")
        support_measurement_dim = support_dim - self.environment_dim
        self.ablation = ablation
        self.gate_evidence_features = gate_evidence_features
        self.environment_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.environment_dim, hidden),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
        )
        self.support_encoder = torch.nn.Sequential(
            torch.nn.Linear(support_measurement_dim, hidden),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
        )
        self.query_encoder = torch.nn.Sequential(
            torch.nn.Linear(4, hidden),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
        )
        self.local_attention = torch.nn.MultiheadAttention(
            hidden,
            attention_heads,
            batch_first=True,
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden * 4, hidden * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden * 2, hidden),
            torch.nn.GELU(),
        )
        self.heads = MultiTaskHeads(hidden, far_count, near_angle_count, near_range_count)
        gate_input_dim = hidden * 4 + (4 if gate_evidence_features else 0)
        self.gates = torch.nn.ModuleDict(
            {
                name: torch.nn.Sequential(
                    torch.nn.Linear(gate_input_dim, hidden),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden, 1),
                )
                for name in ("rss", *TASK_NAMES)
            }
        )

    @staticmethod
    def _gate_evidence(
        prediction: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.ndim == 2:
            difference = prediction - prior
            return torch.stack(
                [
                    torch.tanh(prediction),
                    torch.tanh(prior),
                    torch.tanh(torch.abs(difference)),
                    torch.tanh(prediction * prior),
                ],
                dim=-1,
            )
        neural_probability = torch.softmax(prediction, dim=-1)
        prior_probability = torch.softmax(prior, dim=-1)
        neural_confidence = torch.max(neural_probability, dim=-1).values
        prior_confidence = torch.max(prior_probability, dim=-1).values
        agreement = torch.sum(neural_probability * prior_probability, dim=-1)
        disagreement = 0.5 * torch.sum(
            torch.abs(neural_probability - prior_probability),
            dim=-1,
        )
        return torch.stack(
            [neural_confidence, prior_confidence, agreement, disagreement],
            dim=-1,
        )

    def forward(
        self,
        support_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        local_support_indices: torch.Tensor,
        local_prior: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        support_environment = support_tokens[
            ...,
            3 : 3 + self.environment_dim,
        ]
        support_measurement = torch.cat(
            [
                support_tokens[..., :3],
                support_tokens[..., 3 + self.environment_dim :],
            ],
            dim=-1,
        )
        query_environment = query_tokens[..., 3 : 3 + self.environment_dim]
        query_geometry = torch.cat([query_tokens[..., :3], query_tokens[..., -1:]], dim=-1)
        support = self.support_encoder(support_measurement)
        support_environment_feature = self.environment_encoder(support_environment)
        query_environment_feature = self.environment_encoder(query_environment)
        if self.ablation == "no_environment":
            support_environment_feature = torch.zeros_like(support_environment_feature)
            query_environment_feature = torch.zeros_like(query_environment_feature)
        support = support + support_environment_feature
        query_geometry_feature = self.query_encoder(query_geometry)
        query = query_geometry_feature + query_environment_feature
        global_context = support.mean(dim=1, keepdim=True).expand(-1, query.shape[1], -1)
        if self.ablation == "no_global":
            global_context = torch.zeros_like(global_context)
        batch = torch.arange(support.shape[0], device=support.device)[:, None, None]
        neighbours = support[batch, local_support_indices]
        local_context, _ = self.local_attention(
            query.unsqueeze(2).flatten(0, 1),
            neighbours.flatten(0, 1),
            neighbours.flatten(0, 1),
            need_weights=False,
        )
        local_context = local_context[:, 0].reshape_as(query)
        if self.ablation == "no_local_attention":
            local_context = torch.zeros_like(local_context)
        joint = torch.cat(
            [
                query_geometry_feature,
                query_environment_feature,
                local_context,
                global_context,
            ],
            dim=-1,
        )
        neural_features = self.fusion(joint)
        neural = self.heads(neural_features)
        result: dict[str, torch.Tensor] = {}
        gate_values: dict[str, torch.Tensor] = {}
        for name, prediction in neural.items():
            if self.ablation == "fixed_gate":
                gamma = torch.full_like(self.gates[name](joint), 0.5)
            else:
                gate_input = joint
                if self.gate_evidence_features:
                    gate_input = torch.cat(
                        [joint, self._gate_evidence(prediction, local_prior[name])],
                        dim=-1,
                    )
                gamma = torch.sigmoid(self.gates[name](gate_input))
            prior = local_prior[name]
            if prediction.ndim == 2:
                gamma = gamma.squeeze(-1)
            if self.ablation == "no_local_prior":
                result[name] = prediction
            else:
                result[name] = gamma * prediction + (1.0 - gamma) * prior
            result[f"neural_{name}"] = prediction
            result[f"local_prior_{name}"] = prior
            gate_values[name] = gamma.squeeze(-1)
        result["gates"] = gate_values
        return result
