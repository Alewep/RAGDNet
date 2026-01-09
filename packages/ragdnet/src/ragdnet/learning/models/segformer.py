from torch.nn import CrossEntropyLoss
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from torchmetrics.classification import MulticlassJaccardIndex
import torch.nn.functional as F
import torch
import pytorch_lightning as pl

"""
WARNING : In this setup we systematically observe NaN gradients at the very first training step
(first batch of the first epoch). As a workaround, on_after_backward() sanitizes all
gradients with torch.nan_to_num(), which prevents NaNs from corrupting the weights
and allows the model to train correctly despite this numerical issue.
"""


configs = {
    "segformer_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "segformer_b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "segformer_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "segformer_b3": "nvidia/segformer-b3-finetuned-ade-512-512",
    "segformer_b4": "nvidia/segformer-b4-finetuned-ade-512-512",
    "segformer_b5": "nvidia/segformer-b5-finetuned-ade-640-640",
}


class SegFormerScratch(pl.LightningModule):
    def __init__(self, name="segformer_b0", num_classes=4, lr=1e-3, ignore_index=-1,use_pretrained=False):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lr = lr

        if use_pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                configs[name],
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            print('[INFO] No petrained mode, you can activate it with --pretrained.')
            config = SegformerConfig.from_pretrained(configs[name])
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)

        self.loss_fn = CrossEntropyLoss(ignore_index=ignore_index)

        self.train_miou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="macro",
            ignore_index=ignore_index,
        )
        self.val_miou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="macro",
            ignore_index=ignore_index,
        )

        self.val_iou_per_class = MulticlassJaccardIndex(
            num_classes=num_classes,
            average=None,
            ignore_index=ignore_index,
        )

    def forward(self, x):
        return self.model(pixel_values=x).logits

    def _shared_step(self, batch, stage: str):
        images, masks = batch

        logits = self(images)
        H, W = images.shape[-2], images.shape[-1]

        logits = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        loss = self.loss_fn(logits, masks)
        preds = torch.argmax(logits, dim=1)  # (B,H,W)

        if stage == "train":
            self.train_miou.update(preds, masks)
            self.log(
                "train_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=masks.numel(),
            )
            self.log(
                "train_mIoU",
                self.train_miou,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        else:  # val
            self.val_miou.update(preds, masks)
            self.val_iou_per_class.update(preds, masks)

            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=masks.numel(),
            )
            self.log(
                "val_mIoU",
                self.val_miou,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.val_iou_per_class.reset()
            return

        iou_pc = self.val_iou_per_class.compute()
        for i in range(iou_pc.numel()):
            if torch.isnan(iou_pc[i]):
                continue
            self.log(
                f"val_iou_class_{i}",
                iou_pc[i],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        self.val_iou_per_class.reset()
    
    def on_after_backward(self):
        """Replace NaN/Inf gradients by 0 and print how many parameters were sanitized."""

        reset_count = 0
        for name, p in self.named_parameters():
            if p.grad is None:
                continue

            g = p.grad
            # Check if there is any non-finite gradient
            if not torch.isfinite(g).all():
                reset_count += 1
                # Replace NaN, +Inf, -Inf by 0.0
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                p.grad = g

        if reset_count > 0:
            print(
                f"[Grad cleanup] global_step={self.global_step} | "
                f"sanitized gradients for {reset_count} parameter tensors"
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
