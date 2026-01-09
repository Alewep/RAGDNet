import argparse
import os

from ragdnet.learning.config import RunConfig, parse_args_to_config
import ragdnet.learning.models.gnn as gnn_models
import ragdnet.learning.models.mlp as mlp_models
from ragdnet.learning.datasets.graph_dataset import GraphDrawingDataset
import pytorch_lightning as pl
from ragdnet.learning.train import kfold


def get_gnn_class(name: str):
    try:
        cls = getattr(gnn_models, name)
    except AttributeError as e:

        try:
            cls = getattr(mlp_models, name)

        except AttributeError as e:
            raise ValueError(
                f"Class {name!r} not found"
            ) from e
    

    if not isinstance(cls, type):
        raise TypeError(f"{name!r} is not a class in this module")

    return cls


def create_gnn(name: str, *args, **kwargs):
    cls = get_gnn_class(name)
    return cls(*args, **kwargs)


def build_model_from_config(cfg: RunConfig) -> pl.LightningModule:
    ModelCls = get_gnn_class(cfg.model_name)
    model = ModelCls(
        num_classes=cfg.num_classes,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_pipeline_path",
        type=str,
        required=True,
        help="Path to the pipeline configuration file used to build the dataset.",
    )
    parser.add_argument(
        "--pipeline_name",
        type=str,
        default="default",
        help="Name of the pipeline defined in the configuration file.",
    )

    parser.add_argument(
        "--num_wokers",
        type=int,
        default=5,
    )

    cfg = parse_args_to_config(parser)
    args = parser.parse_args()
    
   
    dataset = GraphDrawingDataset(
        root=cfg.data_path,
        config_path=args.config_pipeline_path,
        num_workers=args.num_wokers,
        num_classes=cfg.num_classes

    )

    basename_cfg = os.path.basename(args.config_pipeline_path).split('.')[0]
    cfg.data_path = os.path.join(cfg.data_path, basename_cfg)


    results = kfold.train_kfold(
        model=build_model_from_config(cfg),
        dataset=dataset,
        cfg=cfg
    )

    print("\n=== K-fold summary ===")
    for res in results:
        print(
            f"Fold {res['fold']} -> best checkpoint: {res['best_checkpoint']}, "
            f"log dir: {res['log_dir']}"
        )


if __name__ == "__main__":
    main()
