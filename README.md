# RAGDNet: A Region-Adjacency Graph for Semantic Segmentation of Mechanical Drawings Using Graph Convolutional Networks

Reference implementation of **RAGDNet**, a region-adjacency-graph formulation for semantic segmentation of mechanical drawings, providing reproducible **image→graph** pipelines and training/inference utilities for **GNN/MLP** models (with pixel-based baselines).

## Features
- DXF → images (optional rendering/colorization).
- Images → graphs (RAGDNet pipeline).
- PyTorch Lightning training with k-fold support.
- Inference + visualization scripts in `scripts/`.

## Layout
- `packages/ragdnet/configs/` — TOML configs (dxf2img, ragdnet).
- `packages/ragdnet/src/ragdnet/pipelines/` — `xpipe` pipelines.
- `packages/ragdnet/src/ragdnet/learning/` — models, datasets, training.
- `packages/ragdnet/data/` — CAD fonts (SHX/TTF) for DXF→image.
- `scripts/` — inference + visualization.

## Requirements
- Python ≥ 3.12 (recommended: `uv`).
- CUDA GPU recommended (CPU works).

## Setup
```bash
git clone <repo>
cd ragdnet
uv venv
uv sync
uv run ezdxf --fonts
````

## Generate data

### DXF → images

```bash
uv run xpipe dxf2img \
  -c packages/ragdnet/configs/dxf_to_image/generated_300dpi.toml \
  -i "packages/ragdnet/data/test/dxf/*.dxf" \
  -o outputs/images
```

### Images → graphs

```bash
uv run xpipe ragdnet \
  -c packages/ragdnet/configs/image_to_graph/ragdnet_r60_130_k1.toml \
  -i "outputs/images/*.png" \
  -o outputs/graphs
```

## Train (graphs → GNN/MLP)

```bash
uv run train_gnn \
  --data_path .\dataset\data\  \
  --num_classes 4 \
  --model_name "MLP_L" \
  --config_pipeline_path "packages/ragdnet/configs/image_to_graph/ragdnet_r60_130_k1.toml"
```

## Train (vision baselines: U-Net / SegFormer-B0)

The following command trains pixel-based baselines (e.g. **U-Net** and **SegFormer-B0**):

```bash
uv run train_vision --data_path .\dataset\data\raw\ --model_name "segformer_b0" --lr 1e-4 --num_classes 4 --pretrained
```

## Monitor

```bash
uv run tensorboard --logdir train_logs/logs
```

## Citation

```bibtex
@misc{monnierweil_ragdnet,
  title   = {RAGDNet: A Region-Adjacency Graph for Semantic Segmentation of Mechanical Drawings Using Graph Convolutional Networks},
  author  = {Alexandre Monnier Weil and Nicolas Hili and Yves Ledru},
  note    = {Kaizen Solutions; Univ. Grenoble Alpes, CNRS, Grenoble INP, LIG},
}
```
