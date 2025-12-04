import torch
import re
from typing import List
from transformers import BitsAndBytesConfig, AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

class Qwen3VLRefiner:
    """
    Handles the Vision-Language Model (Qwen3-VL) for refining coordinates based on text queries.
    """
    def __init__(self, model_name: str = "unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit" , device: str = "cuda:0"):

        self.device = device
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )

        # Optimization: 512 tokens (~780x780 px)
        min_pixels = 264 * 28 * 28    # ~784px
        max_pixels = 1536 * 28 * 28   # ~2048px - enables multi-crop for huge images

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": device}
        )

    def refine_coordinates(self, image_path: str, query: str, yolo_coords: List[List[float]]) -> List[List[float]]:
        """
        Fast coordinate refinement with minimal token usage.
        VLM performs semantic understanding and spatial filtering.
        """
        
        # Create compact coordinate representation
        coords_compact = "\n".join([
            f"{i}: x={c[0]:.3f} y={c[1]:.3f} w={c[2]:.3f} h={c[3]:.3f}"
            for i, c in enumerate(yolo_coords)
        ])
        
        # Concise but clear prompt
        prompt = (
            f"Coordinate system: x∈[0,1] left→right, y∈[0,1] top→bottom\n\n"
            f"Detected objects:\n{coords_compact}\n\n"
            f"Task: {query}\n\n"
            f"Spatial rules:\n"
            f"• rightmost = max x value\n"
            f"• leftmost = min x value\n"
            f"• topmost = min y value\n"
            f"• bottommost = max y value\n\n"
            f"Analyze the image, identify which detections match the query.\n"
            f"Output ONLY the matching detection numbers (comma-separated, e.g., '0,3,7'):\n"
            f"strictly adhere to the query. Donot Output any kind of conversational text or explanation."
        )
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]
        }]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,  # Enough for indices, not for explanations
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        
        # Parse indices - look for numbers in the output
        indices = re.findall(r'\d+', output_text)
        matching_indices = [int(idx) for idx in indices if int(idx) < len(yolo_coords)]
        
        matching_indices = list(dict.fromkeys(matching_indices))
        
        result = [yolo_coords[i] for i in matching_indices]
        
        if not result:
            result = []
        
        return result
