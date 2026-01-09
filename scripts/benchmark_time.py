import argparse
import importlib
import time
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch

from ragdnet.pipelines.factory import create_pipeline
from ragdnet.pipelines.image_to_graph.ragd.runner import RagdnetPipeline
from ragdnet.learning.datasets.graph_dataset import nx_to_pointdata


def load_toml(path: str) -> Dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
        with open(path, "rb") as f:
            return tomllib.load(f)
    except ModuleNotFoundError:
        import tomli  # pip install tomli
        with open(path, "rb") as f:
            return tomli.load(f)


def import_class(dotted: str):
    if ":" not in dotted:
        raise ValueError(f"Invalid class path '{dotted}'. Expected 'module:ClassName'.")
    module_name, class_name = dotted.split(":", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def list_images(root_dir: str) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    root = Path(root_dir)
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files


def maybe_cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def summarize(times: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(times.mean()),
        "std": float(times.std()),
        "p50": float(np.percentile(times, 50)),
        "p90": float(np.percentile(times, 90)),
    }


def load_lightning_model(ckpt: str, class_path: str, device: torch.device) -> torch.nn.Module:
    cls = import_class(class_path)
    model = cls.load_from_checkpoint(ckpt)
    model.to(device).eval()
    return model


def get_model_from_pipeline(pipe: Any):
    if hasattr(pipe, "model") and pipe.model is not None:
        return pipe.model
    if hasattr(pipe, "runner") and hasattr(pipe.runner, "model"):
        return pipe.runner.model
    return None


def image_to_tensor_binary_3ch(path: Path, device: torch.device) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))

    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    x = torch.from_numpy(bin_img).float().unsqueeze(0)  # (1, H, W)
    x = x.repeat(3, 1, 1)                               # (3, H, W)
    x = x.unsqueeze(0).to(device)                       # (1, 3, H, W)
    return x


def benchmark_image_model(
    model: torch.nn.Module,
    image_paths: List[Path],
    device: torch.device,
    label: str,
) -> np.ndarray:
    times: List[float] = []

    with torch.inference_mode():
        for i, p in enumerate(image_paths, start=1):
            x = image_to_tensor_binary_3ch(p, device)
            t0 = time.perf_counter()
            _ = model(x)
            maybe_cuda_sync(device)
            t1 = time.perf_counter()

            dt = t1 - t0
            times.append(dt)
            print(f"[{label}] {i}/{len(image_paths)} - {p.name} {dt:.4f}s")

    return np.array(times, dtype=np.float64)


def process_one_image_gnn(
    pipe: RagdnetPipeline,
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
) -> Tuple[float, float]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(str(image_path))

    t0 = time.perf_counter()

    graph = pipe.run(img, verbose=False)
    pyg_graph = nx_to_pointdata(graph)
    data = pyg_graph.to(device)

    t1 = time.perf_counter()
    with torch.inference_mode():
        out = model(data)
        _ = torch.softmax(out, dim=-1).argmax(dim=-1)
        maybe_cuda_sync(device)
    t2 = time.perf_counter()

    return (t2 - t0), (t2 - t1)


def benchmark_gnn(
    pipe: RagdnetPipeline,
    model: torch.nn.Module,
    image_paths: List[Path],
    device: torch.device,
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.to(device).eval()

    total_times: List[float] = []
    model_times: List[float] = []

    for i, p in enumerate(image_paths, start=1):
        total_t, model_t = process_one_image_gnn(pipe, model, p, device)
        total_times.append(total_t)
        model_times.append(model_t)
        print(
            f"[{label}] {i}/{len(image_paths)} - {p.name} "
            f"total={total_t:.4f}s model={model_t:.4f}s"
        )

    return np.array(total_times, dtype=np.float64), np.array(model_times, dtype=np.float64)


def save_summary_csv(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["name", "type", "metric", "n", "mean", "std", "p50", "p90"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to benchmark.toml")
    ap.add_argument("--csv_out", default="", help="Optional: path to save summary CSV")
    args = ap.parse_args()

    cfg = load_toml(args.config)

    device = torch.device(cfg.get("general", {}).get("device", "cpu"))
    max_images = int(cfg.get("general", {}).get("max_images", 0))
    max_images = None if max_images == 0 else max_images

    root_dir = cfg["dataset"]["root_dir"]
    image_paths = list_images(root_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {root_dir}")
    if max_images is not None:
        image_paths = image_paths[:max_images]

    print(f"Found {len(image_paths)} images in {root_dir}")
    print(f"Device: {device}")

    models_cfg = cfg.get("models", [])
    if not models_cfg:
        raise RuntimeError("No [[models]] entries found in config.")

    csv_rows: List[Dict[str, Any]] = []

    for m in models_cfg:
        name = m["name"]
        mtype = m["type"].lower()

        if mtype == "image":
            model = load_lightning_model(m["ckpt"], m["class"], device)
            times = benchmark_image_model(model, image_paths, device, label=name)
            s = summarize(times)

            print(f"\n=== {name} (image, binary->3ch) ===")
            print(f"mean={s['mean']:.4f}s std={s['std']:.4f}s p50={s['p50']:.4f}s p90={s['p90']:.4f}s\n")

            csv_rows.append({
                "name": name,
                "type": "image",
                "metric": "forward",
                "n": len(times),
                **s,
            })

        elif mtype == "gnn":
            pipe = create_pipeline(m["pipeline_cfg"], RagdnetPipeline, "ragdnet.pipelines")
            model = load_lightning_model(m["ckpt"], m["class"], device)

            total_times, model_times = benchmark_gnn(pipe, model, image_paths, device, label=name)

            s_total = summarize(total_times)
            s_model = summarize(model_times)

            print(f"\n=== {name} (gnn) ===")
            print(
                f"total: mean={s_total['mean']:.4f}s std={s_total['std']:.4f}s "
                f"p50={s_total['p50']:.4f}s p90={s_total['p90']:.4f}s"
            )
            print(
                f"model: mean={s_model['mean']:.4f}s std={s_model['std']:.4f}s "
                f"p50={s_model['p50']:.4f}s p90={s_model['p90']:.4f}s\n"
            )

            csv_rows.append({
                "name": name,
                "type": "gnn",
                "metric": "total",
                "n": len(total_times),
                **s_total,
            })
            csv_rows.append({
                "name": name,
                "type": "gnn",
                "metric": "model",
                "n": len(model_times),
                **s_model,
            })

        else:
            raise ValueError(f"Unknown model type '{mtype}' for '{name}'. Use 'image' or 'gnn'.")

    if args.csv_out:
        save_summary_csv(args.csv_out, csv_rows)
        print(f"Saved CSV summary to: {args.csv_out}")


if __name__ == "__main__":
    main()
