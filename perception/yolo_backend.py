#!/usr/bin/env python3
"""YOLO backend used by the participant baseline."""

import os
from pathlib import Path

import numpy as np


def _hide_cuda_from_this_process() -> None:
    """Stop PyTorch/ultralytics from creating a CUDA context next to Server GS."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _gpu_too_tight_for_yolo(torch, min_free_bytes: int = 2 * 1024**3) -> bool:
    """True when the visible GPU cannot hold YOLO on top of Server 3DGS."""
    try:
        free, _total = torch.cuda.mem_get_info(0)
    except Exception:
        return False
    return int(free) < min_free_bytes


class YoloBackend:
    CLASS_NAMES = ["kele"]

    def __init__(self, weights: Path, confidence: float = 0.65, device: str = "auto"):
        self.confidence = confidence
        self.model = None
        self.device = device

        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")

        if str(device).lower() == "cpu":
            _hide_cuda_from_this_process()

        import torch
        from ultralytics import YOLO

        selected_device = self._select_device(torch, device)
        original_load = torch.load

        def compatible_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = compatible_load
        try:
            self.model = YOLO(str(weights)).to(selected_device)
            self.model.model.eval()
        finally:
            torch.load = original_load

        print(f"[YoloBackend] loaded {weights} on {selected_device}")
        if str(selected_device).startswith("cuda"):
            print(
                "[YoloBackend] warning: cuda YOLO shares the GPU with Server GS; "
                "on an 8GB 4060 that often SIGKILLs the Server. Use --device cpu.",
                flush=True,
            )

    @staticmethod
    def _select_device(torch, requested: str):
        requested = requested.lower()
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("YOLO device must be auto, cpu, or cuda")
        if requested == "cpu":
            return torch.device("cpu")
        if not torch.cuda.is_available():
            if requested == "cuda":
                raise RuntimeError("CUDA was requested but is unavailable")
            print("[YoloBackend] CUDA unavailable; using CPU")
            return torch.device("cpu")

        major, minor = torch.cuda.get_device_capability(0)
        capability = major * 10 + minor
        supported = [
            int(arch[3:])
            for arch in torch.cuda.get_arch_list()
            if arch.startswith("sm_")
        ]
        compatible = any(
            arch // 10 == major and arch % 10 <= minor for arch in supported
        )
        if compatible:
            if requested == "auto" and _gpu_too_tight_for_yolo(torch):
                print(
                    "[YoloBackend] GPU VRAM is tight (Server GS=1 is likely using it); "
                    "using CPU so the Server is not OOM-killed"
                )
                return torch.device("cpu")
            return torch.device("cuda:0")
        if requested == "cuda":
            raise RuntimeError(
                f"GPU sm_{capability} is unsupported by this PyTorch build: {supported}"
            )
        print(
            f"[YoloBackend] GPU sm_{capability} is unsupported by this PyTorch "
            f"build ({supported}); using CPU"
        )
        return torch.device("cpu")

    def detect(self, rgb: np.ndarray) -> list[dict]:
        results = self.model(rgb, verbose=False)[0]
        detections = []
        for box in results.boxes:
            confidence = float(box.conf.item())
            if confidence < self.confidence:
                continue
            class_id = int(box.cls.item())
            if class_id >= len(self.CLASS_NAMES):
                continue
            x0, y0, x1, y1 = map(int, box.xyxy[0].cpu().numpy())
            detections.append(
                {
                    "class": self.CLASS_NAMES[class_id],
                    "x": (x0 + x1) // 2,
                    "y": (y0 + y1) // 2,
                    "w": x1 - x0,
                    "h": y1 - y0,
                    "conf": confidence,
                }
            )
        return detections
