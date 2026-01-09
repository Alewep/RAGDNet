# RAGDNet: A Region-Adjacency Graph for Semantic Segmentation of Mechanical Drawings Using Graph Convolutional Networks

Reference implementation of A Region-Adjacency Graph for Semantic Segmentation of Mechanical Drawings Using Graph Convolutional Networks. The project provides a reproducible image to graph pipeline, plus training and inference utilities for GNN/MLP models and pixel-based baselines.

---

## Features

* DXF to image conversion (optional rendering/colorization)
* Image to graph conversion (RAGDNet pipeline)
* PyTorch Lightning training with k-fold support
* Inference and visualization scripts in `scripts/`
* Pixel-based baselines: U-Net and SegFormer-B0/B2

---

## Repository structure

* `packages/ragdnet/configs/`
  TOML configs (dxf2img, ragdnet)

* `packages/ragdnet/src/ragdnet/pipelines/`
  xpipe pipelines

* `packages/ragdnet/src/ragdnet/learning/`
  models, datasets, training

* `packages/ragdnet/data/`
  CAD fonts (SHX/TTF) for DXF to image

* `scripts/`
  inference and visualization

---

## Requirements

* Python 3.12 or newer (uv recommended)
* CUDA GPU recommended (CPU works)

---

## Setup (Linux)

```bash
git clone <repo>
cd ragdnet
uv venv
uv sync
uv run ezdxf --fonts  # ensure DXF fonts (SHX/TTF) are detected and registered
```

---

## Data generation

### DXF to images

```bash
uv run xpipe dxf2img \
  -c packages/ragdnet/configs/dxf_to_image/generated_300dpi.toml \
  -i "packages/ragdnet/data/test/dxf/*.dxf" \
  -o outputs/images
```



---

## Training

### Graph models (GNN / MLP)

```bash
uv run train_gnn \
  --data_path ./dataset/data/ \ # converts images from the raw/ subfolder into graphs and stores them in a cache for reuse
  --num_classes 4 \
  --model_name "MLP_L" \ # or "GAT_L"
  --config_pipeline_path packages/ragdnet/configs/image_to_graph/ragdnet_r60_130_k1.toml
```

### Vision baselines (U-Net / SegFormer-B0/B2)

```bash
uv run train_vision \
  --data_path ./dataset/data/raw/ \  # train directly from raw images
  --model_name "segformer_b0" \ # or "unet", "segformer_b2"
  --lr 1e-4 \
  --num_classes 4
```

---

## Monitoring

```bash
uv run tensorboard --logdir train_logs/logs
```

---

## Citation

```bibtex
@misc{monnierweil_ragdnet,
  title   = {RAGDNet: A Region-Adjacency Graph for Semantic Segmentation of Mechanical Drawings Using Graph Convolutional Networks},
  author  = {Alexandre Monnier Weil and Nicolas Hili and Yves Ledru},
  note    = {Kaizen Solutions; Univ. Grenoble Alpes, CNRS, Grenoble INP, LIG},
}
```
