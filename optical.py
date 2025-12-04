def inference_streaming_optimized(model, tokenizer, image_path, query, max_new_tokens=512):
    """
    Optimized streaming using TextIteratorStreamer (faster and more efficient).
    Returns the generated text without printing.
    """
    # Load and resize image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path.convert("RGB")
    
    max_size = 768
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.LANCZOS)
    
    # Format message
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": query}
        ]}
    ]
    
    input_text = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        tokenize=False
    )
    
    # Tokenize
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(DEVICE)
    
    # Setup streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=30.0
    )
    
    # Generation kwargs
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Start generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Collect generated text without printing
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
    
    thread.join()
    
    return generated_text.strip()