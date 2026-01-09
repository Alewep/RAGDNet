import torch
from torchmetrics import Metric
from torchmetrics.classification import MulticlassJaccardIndex
from torch_geometric.data import Batch, Data

class RAGImg(Metric):
    full_state_update = False

    def __init__(self, num_classes: int, ignore_index: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.miou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="macro",
            ignore_index=-1,
        )
        self.iou_pc = MulticlassJaccardIndex(
            num_classes=num_classes,
            average=None,
            ignore_index=-1,
        )

    def update(self, preds: torch.Tensor, batch: Batch | Data) -> None:
        graphs, node_logits = self._split_batch(batch, preds)
        device = preds.device

        for graph, logits in zip(graphs, node_logits, strict=False):
            pred_img = self._paint_from_src(graph, logits, device=device)   # (H,W) int
            gt_img = torch.as_tensor(graph.img_mask, device=device).long()  # (H,W) int
            
            # Warning : unsqueeze(0) needed
            self.miou.update(pred_img.unsqueeze(0), gt_img.unsqueeze(0))
            self.iou_pc.update(pred_img.unsqueeze(0), gt_img.unsqueeze(0))


    def compute(self):
        return self.miou.compute()

    def compute_per_class_iou(self):
        return self.iou_pc.compute()
    
    def reset(self):
        self.miou.reset()
        self.iou_pc.reset()
        super().reset()

    def _split_batch(self, batch: Batch | Data, preds: torch.Tensor):
        if isinstance(batch, Batch):
            data_list = batch.to_data_list()
            counts = torch.bincount(batch.batch, minlength=len(data_list))
        else:
            data_list = [batch]
            counts = torch.tensor([preds.size(0)], device=preds.device)

        slices, start = [], 0
        for n in counts.tolist():
            slices.append(preds[start:start + n])
            start += n
        return data_list, slices

    def _paint_from_src(self, graph: Data, logits: torch.Tensor, device) -> torch.Tensor:
        labels = torch.argmax(logits, dim=1).to(device)
        H, W = graph.img_mask.shape
        pred = torch.zeros((H, W), dtype=torch.long, device=device)
        src = graph.src
        for node_id, pix in enumerate(src):
            if pix is None or len(pix) == 0:
                continue
            coords = torch.as_tensor(pix, device=device).long()
            rr, cc = coords[:, 0], coords[:, 1]
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            if ok.any():
                pred[rr[ok], cc[ok]] = labels[node_id]
        return pred


