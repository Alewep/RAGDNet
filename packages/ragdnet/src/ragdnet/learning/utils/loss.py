import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeWeightedCrossEntropyLoss(nn.Module):
    """
    Node classification loss where each graph has (roughly) equal weight.

    logits: [N, C]  – predictions for all nodes in the batch
    target: [N]     – node labels
    batch:  [N]     – batch[i] = graph id of node i
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        # Per-node cross-entropy
        per_node_loss = F.cross_entropy(logits, target, reduction="none")  # [N]

        # Number of graphs in the batch
        num_graphs = int(batch.max()) + 1
        graph_sizes = torch.bincount(batch, minlength=num_graphs).clamp_min(1)  # [G]

        # Weight of each node = 1 / size_of_its_graph
        node_weights = 1.0 / graph_sizes[batch].float()  # [N]

        # Weighted average
        loss = (per_node_loss * node_weights).sum() / node_weights.sum().clamp_min(1e-8)
        return loss


