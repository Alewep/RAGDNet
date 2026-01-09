from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.segmentation import MeanIoU
from torchmetrics import ConfusionMatrix
import torch.nn.functional as F
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, base_channels=64):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.bottom = DoubleConv(base_channels * 8, base_channels * 16)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def _upsample_to(self, x, ref):
            return F.interpolate(
                x,
                size=ref.shape[-2:],  # (H_ref, W_ref)
                mode="bilinear",
                align_corners=False,
            )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)              # B, C, H,   W
        x2 = self.enc2(self.pool(x1))  # B,2C, H/2, W/2
        x3 = self.enc3(self.pool(x2))  # B,4C, H/4, W/4
        x4 = self.enc4(self.pool(x3))  # B,8C, H/8, W/8
        xb = self.bottom(self.pool(x4))# B,16C,H/16,W/16 (floors si pas divisible)

        # Decoder
        u4 = self.up4(xb)
        u4 = self._upsample_to(u4, x4)     # force size = x4
        u4 = torch.cat([u4, x4], dim=1)
        u4 = self.dec4(u4)

        u3 = self.up3(u4)
        u3 = self._upsample_to(u3, x3)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.dec3(u3)

        u2 = self.up2(u3)
        u2 = self._upsample_to(u2, x2)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)
        u1 = self._upsample_to(u1, x1)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.dec1(u1)

        logits = self.out_conv(u1)         # B, num_classes, H, W 
        return logits



class UNetScratch(pl.LightningModule):
    def __init__(
        self,
        in_channels=3,
        num_classes=4,
        lr=1e-3,
        ignore_index=-1,
        base_channels=64,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lr = lr

        self.model = UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )

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
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        images, masks = batch

        logits = self(images)  # (B,C,H,W)
        H, W = masks.shape[-2], masks.shape[-1]

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

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

