def apply_lora_with_shim(base_model, adapter_path, device="cuda"):
    """
    Applies a LoRA adapter to an Unsloth FastVisionModel using a shim 
    to ensure compatibility with the PEFT library.
    
    Source: Extracted from 'changing_adapters _nalin.ipynb'
    """
    print(f"[LoRA Manager] Attempting to load adapter from: {adapter_path}")
    
    # 1. Try Normal Load
    try:
        peft_m = PeftModel.from_pretrained(base_model, adapter_path, device_map="auto")
        print("[LoRA Manager] Success: Loaded adapter (normal mode).")
        return peft_m.to(device)
    except Exception as exc:
        print(f"[LoRA Manager] Normal load failed ({exc}). Switching to Shim...")

    # 2. Shim Wrapper (Fixes attribute access issues between Unsloth & PEFT)
    class _Shim(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.model = wrapped
        
        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)
        
        def __getattr__(self, name):
            if name in ("model", "forward", "__getattr__"):
                return super().__getattr__(name)
            return getattr(self.model, name)

    # 3. Try Shim Load
    try:
        shim = _Shim(base_model)
        peft_m = PeftModel.from_pretrained(shim, adapter_path, device_map="auto")
        print("[LoRA Manager] Success: Loaded adapter via Shim.")
        return peft_m.to(device)
    except Exception as exc2:
        print(f"[LoRA Manager] Critical Error: Shim load failed: {exc2}")
        raise RuntimeError("Failed to load LoRA adapter.") from exc2

# ==============================================================================
# 2. SINGLE INFERENCE FUNCTION (Requested)
# ==============================================================================

def run_inference(base_model, tokenizer, image_path, query, lora_adapter_path=None):
    """
    Executes the full inference pipeline:
    1. Loads Image
    2. Applies LoRA Adapter (if provided) using the Shim method
    3. Prepares Inputs using Unsloth's tokenizer
    4. Generates Response
    
    Args:
        base_model: The loaded Unsloth FastVisionModel (should be the 'raw' base model).
        tokenizer: The loaded Unsloth tokenizer.
        image_path (str): Path to the image file.
        query (str): The text prompt/question.
        lora_adapter_path (str, optional): Path to the LoRA adapter folder. 
                                           If None, runs on base model.
    
    Returns:
        str: The generated text response.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"Error loading image: {e}"

    # 2. Apply Adapter (if provided)
    # NOTE: We assume 'base_model' passed here is the CLEAN base model.
    # This function wraps it locally. The returned 'active_model' is the one we use.
    if lora_adapter_path:
        print(f"🔄 Applying LoRA: {lora_adapter_path}")
        active_model = apply_lora_with_shim(base_model, lora_adapter_path, device)
    else:
        active_model = base_model

    # Ensure inference mode
    FastVisionModel.for_inference(active_model)

    # 3. Prepare Inputs (Unsloth specific syntax)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": query}
        ]}
    ]
    
    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    
    inputs = tokenizer(
        [image],
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    # 4. Generate
    # Use torch.no_grad() for efficiency
    with torch.no_grad():
        outputs = active_model.generate(
            **inputs,
            max_new_tokens=128, # Adjust as needed
            use_cache=True,
            temperature=1.5,
            min_p=0.1
        )
    
    # 5. Decode
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    return response.strip()
