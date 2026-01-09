import argparse
from dataclasses import dataclass,asdict
from pathlib import Path

from sympy import true


@dataclass
class RunConfig:
    # data
    data_path:str

    # Loader
    num_workers: int
    batch_size: int
    batching_mode:str | None

    # Model
    num_classes: int | None 
    model_name: str

    # Training
    patience: int
    max_epochs: int
    lr: float
    weight_decay: float
    min_delta:float

    # Save
    save_dir: str

    # Kfold
    kfold_splits:int 
    kfold_seed:int


def config_to_run_name(cfg:RunConfig,exclude_fields:list=["save_dir","num_workers","kfold_splits"]):
    dconfig = asdict(cfg)
    strings = []
    for field in dconfig:
        if not field in exclude_fields:
            string = dconfig[field]
            if isinstance(string,str):
                p = Path(dconfig[field])
                string = p.name
                

            strings.append(str(string))
    
    return "_".join(strings)


def parse_args_to_config(parser=None) -> RunConfig:
    if parser is None:
        parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_path",type=str,required=True)

    # Loader
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--batching_mode", type=str, default=None)

    # Model
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the GNN class in ragdnet.learning.models.gnn (e.g. GAT_S, GAT_B, GS3, ...)")

    # Training
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--min_delta", type=float, default=1e-3)
    
    # Save
    parser.add_argument("--save_dir", type=str, default="train_logs")

    # K-fold
    parser.add_argument("--kfold_splits", type=int, default=5)
    parser.add_argument("--kfold_seed", type=int, default=42)



    args = parser.parse_args()

    cfg = RunConfig(
        data_path=args.data_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        model_name=args.model_name,
        patience=args.patience,
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        min_delta=args.min_delta,
        save_dir=args.save_dir,
        kfold_splits=args.kfold_splits,
        kfold_seed=args.kfold_seed,
        batching_mode=args.batching_mode,

    )

    return cfg



