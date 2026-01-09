from __future__ import annotations

from abc import ABC, abstractmethod

import itertools
from typing import  Optional

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn import GATv2Conv, SAGEConv
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification.f_beta import MulticlassF1Score

from ragdnet.learning.models.pointset import EdgeMLPEncoder, PointSetEncoder
from ragdnet.learning.utils.loss import NodeWeightedCrossEntropyLoss
from ragdnet.learning.utils.metrics import RAGImg

def build_node_metrics(num_classes: int) -> MetricCollection:
    return MetricCollection(
        {
            "acc": MulticlassAccuracy(num_classes=num_classes, average="macro"),
            "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
            "f1_per_class": MulticlassF1Score(num_classes=num_classes, average="none"),
        }
    )


class BaseGNNClassifer(pl.LightningModule, ABC):
    def __init__(
        self,
        *,
        num_classes: int,
        lr: float,
        weight_decay: float,
        point_encoder: PointSetEncoder | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["point_encoder"])

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        self.loss_fn = NodeWeightedCrossEntropyLoss()

        # node-level metrics
        self.train_node = build_node_metrics(num_classes)
        self.val_node   = build_node_metrics(num_classes)

        # RAG metrics
        self.train_rag = RAGImg(num_classes)
        self.val_rag   = RAGImg(num_classes)

        self.point_encoder = point_encoder
        self.node_feat_dim = (
            getattr(point_encoder, "out_dim", None)
            if point_encoder is not None
            else None
        )

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def forward(self, batch: Batch | Data) -> torch.Tensor:
        x = self._forward_node_features(batch)
        return self.forward_graph(
            x,
            batch.edge_index,
            getattr(batch, "batch", None),
            edge_attr=getattr(batch, "edge_attr", None),
        )

    @abstractmethod
    def forward_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits (num_nodes, num_classes)."""

    def _forward_node_features(self, data: Data) -> torch.Tensor:
        if hasattr(data, "x") and data.x is not None:
            return data.x

        if self.point_encoder is None:
            raise RuntimeError("No point_encoder and no node features.")

        points = data.points
        point2node = data.point2node
        num_nodes = int(data.num_nodes)
        out_dim = self.point_encoder.out_dim

        feats = []
        for i in range(num_nodes):
            pts = points[point2node == i]
            if pts.numel() == 0:
                feats.append(
                    torch.zeros(
                        out_dim,
                        device=points.device,
                        dtype=points.dtype,
                    )
                )
            else:
                feats.append(self.point_encoder(pts))
        return torch.stack(feats, dim=0)

    # -------- steps --------

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def _shared_step(self, batch: Batch | Data, stage: str):
        logits = self(batch)
        preds = torch.argmax(logits, dim=1)

        loss = self.loss_fn(
            logits, batch.y, getattr(batch, "batch", None)
        )

        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if stage == "train":
            self.train_node.update(preds, batch.y)
            self.train_rag.update(logits, batch)
            self.log(
                "train_mIoU",
                self.train_rag.miou,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        else:
            self.val_node.update(preds, batch.y)
            self.val_rag.update(logits, batch)
            self.log(
                "val_mIoU",
                self.val_rag.miou,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        
        return loss

    # -------- end --------
    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")
        self.train_node.reset()
        self.train_rag.reset()

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.val_node.reset()
            self.val_rag.reset()
            return

        self._log_epoch_metrics("val")
        self.val_node.reset()
        self.val_rag.reset()

    def _log_epoch_metrics(self, stage: str):
        node = self.train_node if stage == "train" else self.val_node
        rag  = self.train_rag  if stage == "train" else self.val_rag

        # node metrics
        out = node.compute()
        for k, v in out.items():
            if v.ndim == 1:
                for i in range(v.numel()):
                    if torch.isnan(v[i]):
                        continue
                    self.log(
                        f"{stage}_{k}_{i}",
                        v[i],
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )
            else:
                self.log(
                    f"{stage}_{k}",
                    v,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )

        # per-class IoU
        iou_pc = rag.compute_per_class_iou()
        for i in range(iou_pc.numel()):
            if torch.isnan(iou_pc[i]):
                continue
            self.log(
                f"{stage}_iou_class_{i}",
                iou_pc[i],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )


class GATv2Classifier(BaseGNNClassifer):
    def __init__(
        self,
        conv_dims: list[int],
        linear_dims: list[int],
        *,
        edge_in_channels: Optional[int] = 3,
        heads: int = 1,
        dropout: float = 0.0,
        concat: bool = False,
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


        if self.point_encoder is None or self.node_feat_dim is None:
            raise ValueError(
                "GATv2Classifier expects node features to come from a SetPointEncoder. "
                "Provide `point_encoder=...` when instantiating the model."
            )

        self.concat = concat
        self.heads = heads
        self.dropout = dropout

        in_channels = self.node_feat_dim

        self.gats = nn.ModuleList()
        self.norms = nn.ModuleList()

        prev_dim = in_channels
        for d in conv_dims:
            self.gats.append(
                GATv2Conv(
                    in_channels=prev_dim,
                    out_channels=d,
                    heads=heads,
                    edge_dim=edge_in_channels,
                    dropout=dropout,
                    concat=concat,
                )
            )
            out_dim = d * heads if concat else d
            self.norms.append(nn.LayerNorm(out_dim))
            prev_dim = out_dim

        lin_in = [prev_dim, *linear_dims]
        lin_out = [*linear_dims, num_classes]

        self.linears = nn.ModuleList(
            nn.Linear(d_in, d_out)
            for d_in, d_out in zip(lin_in, lin_out, strict=False)
        )

    def forward_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for gat, norm in zip(self.gats, self.norms):
            x = gat(x, edge_index, edge_attr=edge_attr) if edge_attr is not None else gat(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class SageClassifier(BaseGNNClassifer):
    def __init__(
        self,
        conv_dims: list[int],
        linear_dims: list[int],
        *,
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

        if self.point_encoder is None or self.node_feat_dim is None:
            raise ValueError(
                "SageClassifier expects node features to come from a SetPointEncoder. "
                "Provide `point_encoder=...` when instantiating the model."
            )

        in_channels = self.node_feat_dim

        dims = [in_channels, *conv_dims]
        self.convs = nn.ModuleList(
            SAGEConv(d_in, d_out) for d_in, d_out in itertools.pairwise(dims)
        )

        lin_in = [conv_dims[-1], *linear_dims]
        lin_out = [*linear_dims, num_classes]
        self.linears = nn.ModuleList(
            nn.Linear(d_in, d_out)
            for d_in, d_out in zip(lin_in, lin_out, strict=False)
        )

    def forward_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1:
                x = F.relu(x)

        return x


class GAT_L(GATv2Classifier):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("point_encoder", EdgeMLPEncoder())
        kwargs["conv_dims"] = [256, 512]
        kwargs["linear_dims"] = [256, 128]
        kwargs.setdefault("heads", 2)
        kwargs.setdefault("dropout", 0.0)
        kwargs.setdefault("edge_in_channels", 3)
        super().__init__(*args, **kwargs)
