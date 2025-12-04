import torch
import math
from ultralytics import YOLO
from typing import List
from src.lora import LoRALayer

import time
class YOLOLoRADetector:
    """
    Handles loading the YOLO model, injecting LoRA layers, and running initial detection.
    """
    def __init__(self, base_model_path: str, lora_weights_path: str ,device: str = "cuda:1"):
        self.device = torch.device(device)  # ADD THIS
        self.device_str = device
        
        print(f"Loading YOLO base model: {base_model_path}")
        self.model = YOLO(base_model_path)

        print("Injecting LoRA layers...")
        self._inject_lora()

        self._load_lora_weights(lora_weights_path)
        print("✓ YOLO+LoRA model ready")

    def _inject_lora(self):
        """Iterates through model modules and replaces specific conv layers with LoRA wrappers."""
        for name, module in self.model.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                if "cv1" in name or "cv2" in name:
                    parent_name = name.rsplit('.', 1)[0]
                    child_name = name.rsplit('.', 1)[1]
                    parent = self.model.model.get_submodule(parent_name)
                    lora_layer = LoRALayer(module, rank=16, alpha=16)
                    setattr(parent, child_name, lora_layer)

    def _load_lora_weights(self, weights_path):
        """Loads the LoRA state dict onto the modified model."""
        self.model.model.to(self.device)

        lora_state_dict = torch.load(weights_path, map_location=self.device)
        self.model.model.load_state_dict(lora_state_dict, strict=False)

        # Monkey patch fuse to prevent errors during inference
        self.model.model.fuse = lambda verbose=False: self.model.model

    def detect(self, image_path: str, conf_threshold: float = 0.2) -> List[List[float]]:
        """
        Detect objects and return normalized RBox coordinates.
        Returns: List of [cx, cy, w, h, angle]
        """
        results = self.model.predict(
            image_path,
            imgsz=1024,
            conf=conf_threshold,
            iou=0.4,
            verbose=False,
            device=self.device_str
        )
        detections = []
        for r in results:
            img_h, img_w = r.orig_shape

            if r.obb is not None:
                for det in r.obb.xywhr.cpu().numpy():
                    cx, cy, w, h, rot_rad = det

                    # Normalize spatial coordinates
                    norm_cx = float(cx) / img_w
                    norm_cy = float(cy) / img_h
                    norm_w = float(w) / img_w
                    norm_h = float(h) / img_h

                    # Convert rotation to degrees and enforce [-90, 0]
                    angle_deg = math.degrees(rot_rad)
                    while angle_deg > 0:
                        angle_deg -= 90
                        norm_w, norm_h = norm_h, norm_w
                    while angle_deg <= -90:
                        angle_deg += 90
                        norm_w, norm_h = norm_h, norm_w

                    detections.append([
                        float(norm_cx), float(norm_cy), float(norm_w), float(norm_h), float(angle_deg)
                    ])
        detections = [[round(val, 3) for val in coord] for coord in detections]
        return detections
