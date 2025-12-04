import sys
from src.pipeline import YOLOQwenPipeline

# Singleton instance to hold the loaded models
_PIPELINE_INSTANCE = None

def initialize_pipeline(yolo_path: str = "yolo11n-obb.pt", 
                        lora_path: str = "lora_weights.pt",
                        qwen_model: str = "unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit"):
    """
    Loads the models into memory. Run this once at startup.
    """
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is not None:
        print("Pipeline already initialized.")
        return

    try:
        _PIPELINE_INSTANCE = YOLOQwenPipeline(
            yolo_base_path=yolo_path,
            lora_weights_path=lora_path,
            qwen_model_name=qwen_model
        )
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        sys.exit(1)

def get_coordinates(image_path: str, query: str, conf_threshold: float = 0.2):
    """
    Simple interface to get coordinates.
    Returns list of [cx, cy, w, h, angle]
    """
    if _PIPELINE_INSTANCE is None:
        raise RuntimeError("Pipeline not initialized. Call main.initialize_pipeline() first.")

    result = _PIPELINE_INSTANCE.process(image_path, query, conf_threshold)
    return result['final_coordinates']

# Example usage if run directly
if __name__ == "__main__":
    # 1. Setup
    print("Initializing...")
    initialize_pipeline(
        yolo_path="yolo11n-obb.pt", 
        lora_path="lora_weights.pt"
    )
    
    # 2. Run
    img_path = "test_image.png" 
    user_query = "Locate all the airplanes"
    
    try:
        coords = get_coordinates(img_path, user_query)
        print(f"\nFound {len(coords)} objects:")
        for c in coords:
            print(c)
    except Exception as e:
        print(f"Error during execution: {e}")