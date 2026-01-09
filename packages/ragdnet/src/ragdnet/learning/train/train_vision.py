import argparse

from ragdnet.learning.config import RunConfig, parse_args_to_config
from ragdnet.learning.datasets.image_dataset import DrawingDataset
from ragdnet.learning.models.segformer import SegFormerScratch
from ragdnet.learning.models.unet import UNetScratch
from ragdnet.learning.train import kfold
import pytorch_lightning as pl


def build_model_from_config(cfg: RunConfig, args:argparse.Namespace) -> pl.LightningModule:
    if cfg.model_name == "unet":
        return UNetScratch(num_classes=cfg.num_classes, lr=cfg.lr)

    if cfg.model_name.startswith("segformer"):
        return SegFormerScratch(
            cfg.model_name,
            num_classes=cfg.num_classes,
            lr=cfg.lr,
            use_pretrained=args.pretrained,
        )

    raise ValueError(f"Unknown model_name: {cfg.model_name!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use pretrained weights (default: False).",
    )

    cfg = parse_args_to_config(parser)
    args = parser.parse_args()

    dataset = DrawingDataset(cfg.data_path, num_classes=cfg.num_classes)

    #build model
    model = build_model_from_config(cfg,args)

    #update name
    if args.pretrained:
        cfg.model_name += "_pretrained"
        print(cfg.model_name)

    results = kfold.train_kfold(
        cfg=cfg,
        dataset=dataset,
        model=model,
    )

    print("\n=== K-fold summary ===")
    for res in results:
        print(f"Fold {res['fold']} -> best checkpoint: {res['best_checkpoint']}, ")


if __name__ == "__main__":
    main()
