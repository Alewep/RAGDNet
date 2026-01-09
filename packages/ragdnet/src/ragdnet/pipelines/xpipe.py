#!/usr/bin/env python3
"""Main module to run similarity pipelines in parallel."""

import argparse
import concurrent.futures
import glob
import os
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

from ragdnet.pipelines.dxf_to_image.runner import DXFImagePipeline
from ragdnet.pipelines.factory import Pipeline, create_pipeline
from ragdnet.pipelines.image_to_graph.ragd.runner import RagdnetPipeline


# --- Map of string keys → Pipeline classes ---
PIPELINE_MAP: dict[str, type[Pipeline]] = {
    "dxf2img": DXFImagePipeline,
    "ragdnet": RagdnetPipeline,
}


def worker(task_args: tuple) -> None:
    """Instantiate the chosen pipeline and run it on one file."""
    file_path, config_path, pipeline_key, package, output_dir = task_args
    pipeline_class = PIPELINE_MAP[pipeline_key]

    try:
        pipeline: Pipeline = create_pipeline(config_path, pipeline_class, package)
        pipeline.run_io(file_path, output_dir)
    except Exception as exc:
        # Add context without changing external behavior
        raise RuntimeError(
            f"Pipeline '{pipeline_key}' failed for file: {file_path}"
        ) from exc


def safe_glob(pattern: str) -> list[Path]:
    """Safely resolve glob patterns, handling both absolute and relative paths."""
    p = Path(pattern)
    if p.is_absolute():
        return [Path(f) for f in glob.glob(pattern, recursive=True)]  # noqa: PTH207
    return list(Path().glob(pattern))


def main() -> None:
    """Run a chosen pipeline (positional key) over multiple files in parallel."""
    parser = argparse.ArgumentParser(
        description="Run a chosen pipeline (positional key) "
        "over multiple files in parallel."
    )
    parser.add_argument(
        "pipeline", choices=PIPELINE_MAP.keys(), help="Key of the pipeline to execute"
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Path to the config.toml file"
    )
    parser.add_argument(
        "-p",
        "--package",
        default="ragdnet.pipelines",
        help="Package where strategy classes live (default: pipelines)",
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        required=True,
        help="List of files or glob patterns to process",
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel worker processes (default: number of CPUs)",
    )
    args = parser.parse_args()

    # Resolve glob patterns
    all_files: list[str] = []
    for pattern in args.inputs:
        all_files.extend([str(p) for p in safe_glob(pattern)])
    all_files = sorted(set(all_files))
    if not all_files:
        sys.stderr.write("No files found for the given patterns.\n")
        sys.exit(1)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Prepare tasks
    tasks = [
        (file_path, args.config, args.pipeline, args.package, args.output)
        for file_path in all_files
    ]

    # Execute in parallel with progress bar
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, task): task[0] for task in tasks}

        try:
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing files",
                unit="file",
            ):
                file_path = futures[future]
                try:
                    future.result()
                except FileNotFoundError as exc:
                    # Keep your original idea: log and continue
                    tqdm.write(f"[NOT FOUND] {file_path}: {exc}")
                except Exception as exc:
                    # Show error immediately with file context
                    tqdm.write(f"[ERROR] {file_path}: {exc}")

                    # Helpful traceback at the parent side
                    tb = "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )
                    tqdm.write(tb.rstrip())

                    # Preserve original behavior: stop on non-FileNotFoundError
                    raise

        except KeyboardInterrupt:
            tqdm.write("Interrupted by user. Cancelling remaining tasks...")

            # Cancel tasks that have not started yet
            for f in futures:
                f.cancel()

            # Do not keep waiting; request cancellation of pending futures
            executor.shutdown(wait=False, cancel_futures=True)

            # Standard exit code for SIGINT
            raise SystemExit(130)

    tqdm.write("All done!")


if __name__ == "__main__":
    main()
