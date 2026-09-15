import os as _os
import shutil as _shutil

# Fix: Windows lacks symlink privileges; fall back to file copy for HuggingFace cache
_orig_symlink = _os.symlink
def _safe_symlink(src, dst, target_is_directory=False, dir_fd=None):
    try:
        _orig_symlink(src, dst, target_is_directory=target_is_directory, dir_fd=dir_fd)
    except OSError:
        abs_src = _os.path.join(_os.path.dirname(dst), src) if not _os.path.isabs(src) else src
        if _os.path.exists(abs_src) and not _os.path.exists(dst):
            _os.makedirs(_os.path.dirname(dst), exist_ok=True)
            _shutil.copy2(abs_src, dst)
_os.symlink = _safe_symlink

import numpy as np
import torch
from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation


class DepthEstimator:
    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Base-hf",
        device: str = "cuda",
    ):
        self.device = device
        torch.backends.cudnn.benchmark = True
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        self.model = (
            DepthAnythingForDepthEstimation.from_pretrained(model_id)
            .to(device)
            .eval()
            .half()
        )

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        """Run depth inference on an RGB frame. Returns float32 H×W depth array."""
        h, w = rgb.shape[:2]
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.inference_mode(), torch.amp.autocast(self.device, dtype=torch.float16):
            out = self.model(**inputs)
            pred = out.predicted_depth
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)
            pred = torch.nn.functional.interpolate(
                pred, size=(h, w), mode="bilinear", align_corners=False
            )
            depth = pred[0, 0].float().cpu().numpy().astype(np.float32)
        return depth
