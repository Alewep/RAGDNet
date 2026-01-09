from typing import List, Dict, Optional
import copy
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ragdnet.learning.config import RunConfig, config_to_run_name
from ragdnet.learning.datasets.base import AlignedImageDatasetBase
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from torch_geometric.data import Data as GeoData
from torch_geometric.loader import DataLoader as GeoDataLoader


pl.seed_everything(42,workers=True)

VAL_BATCH_SIZE = 1

class PrintEarlyStoppingState(pl.callbacks.Callback):
    def on_validation_end(self, trainer, pl_module):
        for cb in trainer.callbacks:
            if isinstance(cb, pl.callbacks.EarlyStopping):
                state = cb.state_dict()

                wait = state.get("wait_count", None)
                patience = state.get("patience", None)
                best = state.get("best_score", None)

                print(f"[EarlyStopping] wait_count={wait}/{patience}, best_score={best}")
                break


def _ckpt_was_early_stopped(ckpt_path: str) -> bool:
    """Return True if the checkpoint corresponds to a run stopped by EarlyStopping."""
    try:
        state = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"Could not load checkpoint {ckpt_path}: {e}")
        return False

    callbacks_state = state.get("callbacks", {})
    print("--- Callbacks state ---")
    print(callbacks_state)
    for name, cb_state in callbacks_state.items():
        # The callback name usually contains 'EarlyStopping'
        if "EarlyStopping" in str(name) or "earlystopping" in str(name).lower():
            stopped_epoch = cb_state.get("stopped_epoch", 0)
            if isinstance(stopped_epoch, torch.Tensor):
                stopped_epoch = stopped_epoch.item()
            return stopped_epoch > 0
    return False


def train_one_fold(
    model: pl.LightningModule,
    train_dataset: Dataset,
    val_dataset: Dataset,
    cfg: RunConfig,
    *,
    fold_idx: int,
    base_run_name: str,
    extra_callbacks: Optional[List[pl.callbacks.Callback]] = None,
) -> Dict:
    """
    Train one fold with TensorBoard logging and early stopping.
    Compatible with both standard and PyG datasets.
    """

    # Select the appropriate DataLoader depending on the dataset element type
    if GeoData is not None and isinstance(train_dataset[0], GeoData):
        loader_cls = GeoDataLoader
   
    else:
        loader_cls = DataLoader
    
    train_loader = loader_cls(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader = loader_cls(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Create output directories
    os.makedirs(cfg.save_dir, exist_ok=True)
    log_dir = os.path.join(cfg.save_dir, "logs")
    ckpt_dir = os.path.join(cfg.save_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    run_name = f"{base_run_name}__fold={fold_idx}"

    dirpath = os.path.join(ckpt_dir, run_name)

    # Reload last model
    resume_ckpt = os.path.join(dirpath, "last.ckpt")

    if os.path.exists(resume_ckpt):
        print(f"LAST_VERSION in {resume_ckpt}")
        if _ckpt_was_early_stopped(resume_ckpt):
            print("[WARNING] Model was already early stopped.")
            return None

        old_ckpt = os.path.join(dirpath, "last_old.ckpt")
        if os.path.exists(old_ckpt):
            print("Delete old last model")
            os.remove(old_ckpt)
        os.rename(resume_ckpt, old_ckpt)
        resume_ckpt = old_ckpt
    else:
        resume_ckpt = None

    checkpoint_cb = ModelCheckpoint(
        dirpath=dirpath,
        filename="{epoch:02d}-{val_mIoU:.4f}",
        save_top_k=1,
        monitor="val_mIoU",
        mode="max",
        save_last=False,
    )

    checkpoint_last = ModelCheckpoint(
        dirpath=dirpath,
        filename="last",
        monitor=None,     
        save_top_k=1,      
    )

    early_stopping_cb = EarlyStopping(
        monitor="val_mIoU",
        mode="max",
        patience=cfg.patience,
        min_delta=cfg.min_delta
    )


    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=run_name,
    )

    callbacks: List[pl.callbacks.Callback] = [
        checkpoint_cb,
        checkpoint_last,
        early_stopping_cb,
        PrintEarlyStoppingState(),
    ]
    if extra_callbacks:
        callbacks += list(extra_callbacks)

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_ckpt,weights_only=False)


    return {
        "fold": fold_idx,
        "run_name": run_name,
        "best_checkpoint": checkpoint_cb.best_model_path,
        "best_score": early_stopping_cb.best_score.item()
    }



def train_kfold(
    cfg: RunConfig,
    dataset: AlignedImageDatasetBase,
    model: pl.LightningModule,
) -> List[Dict]:
    """
    Generic K-Fold training loop for any PyTorch Lightning model.
    If the dataset implements `set_train(bool)`, training/validation transforms
    are automatically set for each fold.
    """

    kf = KFold(
        n_splits=cfg.kfold_splits,
        shuffle=True,
        random_state=cfg.kfold_seed,
    )

    base_run_name = config_to_run_name(cfg)
    results: List[Dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n===== FOLD {fold_idx + 1}/{cfg.kfold_splits} =====")
        print(f"Train size: {len(train_idx)} | Val size: {len(val_idx)}")

        model_fold = copy.deepcopy(model)

        # Create two independent copies of the dataset
        train_dataset:AlignedImageDatasetBase = copy.deepcopy(dataset)
        val_dataset:AlignedImageDatasetBase = copy.deepcopy(dataset)
   
        # If the dataset supports train/eval mode switching, apply it
        if hasattr(train_dataset, "set_train"):
            train_dataset.set_train(True)
        if hasattr(val_dataset, "set_train"):
            val_dataset.set_train(False)

        # Apply KFold splits on both copies
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)


        val_paths = [val_subset.dataset.image_paths[i] for i in val_subset.indices]
        print(len(val_paths))
        print([os.path.basename(p) for p in val_paths])  # juste les noms de fichiers


        res = train_one_fold(
            model=model_fold,
            train_dataset=train_subset,
            val_dataset=val_subset,
            cfg=cfg,
            fold_idx=fold_idx,
            base_run_name=base_run_name,
        )
        results.append(res)
       

    return results
