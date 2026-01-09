import itertools
import torch
from torch import nn
from torch.nn import functional as F

from ragdnet.learning.models.gnn import BaseGNNClassifer
from ragdnet.learning.models.pointset import EdgeMLPEncoder


class MLPClassifier(BaseGNNClassifer):
    """
    Baseline MLP node-level classifier.
    Ignores graph structure (edge_index/edge_attr).

    Node features come from:
        - data.x if present
        - otherwise point_encoder

    This version adds LayerNorm on each hidden layer
    to mimic the normalization rhythm of GAT_L.
    """

    def __init__(
        self,
        linear_dims: list[int],
        *,
        dropout: float = 0.0,
        num_classes: int,
        lr: float,
        weight_decay: float,
        **base_kwargs: dict,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            **base_kwargs,
        )
        self.save_hyperparameters(ignore=["point_encoder"])

        # Keep the same expectation as other models for consistency
        if self.point_encoder is None or self.node_feat_dim is None:
            raise ValueError(
                "MLPClassifier expects node features to come from a PointSetEncoder. "
                "Provide `point_encoder=...` when instantiating the model."
            )

        self.dropout = dropout

        in_channels = self.node_feat_dim
        dims = [in_channels, *linear_dims, num_classes]

        self.linears = nn.ModuleList(
            nn.Linear(d_in, d_out) for d_in, d_out in itertools.pairwise(dims)
        )

        # LayerNorm only for hidden layers
        self.norms = nn.ModuleList(nn.LayerNorm(d) for d in linear_dims)

    def forward_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for i, lin in enumerate(self.linears):
            x = lin(x)

            if i < len(self.linears) - 1:
                x = self.norms[i](x)
                x = F.relu(x, inplace=True)

                if self.dropout and self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class MLP_L(MLPClassifier):
    """
    MLP baseline roughly matching GAT_L trainable params
    with a GAT-like capacity profile.
    """
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("point_encoder", EdgeMLPEncoder())
        kwargs["linear_dims"] = [512, 1024, 256, 128]
        kwargs.setdefault("dropout", 0.0)
        super().__init__(*args, **kwargs)
