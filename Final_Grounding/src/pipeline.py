from src.detector import YOLOLoRADetector
from src.refiner import Qwen3VLRefiner
from pathlib import Path
from typing import Dict, Union

class YOLOQwenPipeline:
    """
    Orchestrator class that manages the flow between YOLO and Qwen.
    """
    def __init__(self, yolo_base_path: str, lora_weights_path: str,
                 qwen_model_name: str = "unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit" , device=1):
        self.device=device
        self.yolo_detector = YOLOLoRADetector(yolo_base_path, lora_weights_path, device=self.device)
        self.qwen_refiner = Qwen3VLRefiner(qwen_model_name, device=self.device)
        print("✓ Pipeline initialized")

    def process(self, image_path: str, query: str, conf_threshold: float = 0.3) -> Dict:

        yolo_coords = self.yolo_detector.detect(image_path, conf_threshold)

        if not yolo_coords:
            return {'final_coordinates': [], 'num_detections': 0}

        final_coords = self.qwen_refiner.refine_coordinates(image_path, query, yolo_coords)

        return {
            'yolo_coordinates': yolo_coords,
            'final_coordinates': final_coords,
            'num_detections': len(final_coords)
        }
