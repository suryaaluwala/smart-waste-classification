from __future__ import annotations

import importlib.metadata
import importlib.util

PACKAGES = ("torch", "torchvision", "torch-directml", "streamlit", "scikit-learn", "matplotlib", "Pillow")


def main() -> None:
    for package in PACKAGES:
        try:
            print(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            continue

    try:
        import torch
    except ImportError:
        print("PyTorch could not be imported.")
        raise SystemExit(1)

    if torch.cuda.is_available():
        print("Selected device: CUDA")
    elif importlib.util.find_spec("torch_directml") is not None:
        import torch_directml
        device = torch_directml.device()
        name = torch_directml.device_name(0)
        print(f"Selected device: DirectML ({name}) [{device}]")
        print("DirectML uses a separate, older preview stack; see requirements-directml.txt.")
    else:
        print("Selected device: CPU")
        print("Training will run on the CPU. This dataset is small enough for a learning project.")


if __name__ == "__main__":
    main()
