# Model 2041 - Vision-Language Grounding System

A comprehensive vision-language AI system for object detection, grounding, and visual question answering (VQA) using YOLO, Qwen3-VL, and custom LoRA adapters.
---

## 🎯 Overview

This project implements a multi-modal AI system that combines:
- *Object Detection*: YOLO11n-OBB with custom LoRA weights for oriented bounding box detection
- *Vision-Language Understanding*: Qwen3-VL-32B for semantic understanding and coordinate refinement
- *Image Classification*: Automatic classification of SAR, Optical, and FCC images
- *Query Classification*: Intelligent routing of queries to VQA or Grounding tasks
- *RESTful API*: Flask-based API server for production deployment

---

## 📁 Repository Structure


model_2041-main/
│
├── Final_Grounding/              # Main grounding pipeline module
│   ├── main.py                   # Entry point for grounding pipeline
│   ├── requirements.txt          # Dependencies for grounding module
│   ├── yolo11n-obb.pt           # YOLO base model weights (Git LFS)
│   ├── lora_weights.pt          # LoRA weights for YOLO (Git LFS)
│   └── src/
│       ├── detector.py          # YOLO+LoRA detector implementation
│       ├── lora.py              # LoRA layer implementation
│       ├── pipeline.py          # Main pipeline orchestrator
│       └── refiner.py           # Qwen3-VL coordinate refiner
│
├── testing/                      # API server and inference pipeline
│   ├── api_server.py            # Flask API server (main entry point)
│   ├── inference.py             # Main inference pipeline orchestrator
│   ├── model_loading.py         # Model loading utilities
│   ├── classifier.py            # Image type classifier (SAR/Optical/FCC)
│   ├── query_classifier.py      # Query type classifier (VQA/Grounding)
│   ├── vqa_inference.py         # Visual Question Answering module
│   ├── geo_ground.py            # Geographic grounding module
│   ├── fcc.py                   # FCC-specific processing
│   ├── adapter_changing.py      # LoRA adapter switching utilities
│   ├── requirements_api.txt     # API server dependencies
│   ├── README_DEPLOY.md         # Deployment documentation
│   ├── DEPLOYMENT_CHECKLIST.md  # Deployment checklist
│   ├── SAR_LORA_ADAPTER/        # SAR-specific LoRA adapter (Git LFS)
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   └── Qwen_for_Optical_2/      # Optical-specific LoRA adapter (Git LFS)
│       ├── adapter_config.json
│       └── adapter_model.safetensors
│
├── temp/                         # Development and training files
│   ├── classifiers/
│   │   ├── final_model_all_classes.keras  # Image classifier model
│   │   └── query_classifer.py            # Query classifier implementation
│   ├── fcc model/
│   │   └── fcc best model.pt              # FCC model weights
│   ├── fcc.ipynb                 # FCC training notebook
│   ├── image_classifier.ipynb    # Image classifier training notebook
│   ├── Multimodal.ipynb         # Multimodal training notebook
│   └── qwen_lora.py             # LoRA adapter utilities
│
├── optical.py                    # Optical image inference utility
├── Dockerfile                    # Docker container configuration
├── .gitattributes               # Git LFS configuration
└── README.md                    # This file


---

## 🔄 File Flow & Architecture

### Main Pipeline Flow


User Input (Image + Query)
    │
    ├─→ Image Classification (SAR/Optical/FCC)
    │   └─→ classifier.py → Select appropriate LoRA adapter
    │
    ├─→ Query Classification (VQA/Grounding)
    │   └─→ query_classifier.py → Route to appropriate handler
    │
    └─→ Task Execution
        │
        ├─→ VQA Path
        │   └─→ vqa_inference.py
        │       └─→ Qwen3-VL with image-specific LoRA
        │           └─→ Text response
        │
        └─→ Grounding Path
            └─→ geo_ground.py / Final_Grounding/
                ├─→ YOLO Detection (detector.py)
                │   └─→ Initial bounding boxes with LoRA
                │
                └─→ Qwen3-VL Refinement (refiner.py)
                    └─→ Refined coordinates matching query
                        └─→ Final coordinates [cx, cy, w, h, angle]


### Detailed Component Flow

#### 1. *API Server Flow* (testing/api_server.py)

POST /infer
    │
    ├─→ Download image (if URL provided)
    ├─→ Initialize InferencePipeline (singleton)
    │   └─→ Load all models once (cached in memory)
    │
    └─→ pipeline.run(image_path, query)
        │
        ├─→ Classify Image Type
        │   └─→ Load appropriate LoRA adapter
        │
        ├─→ Classify Query Type
        │   └─→ Determine task (VQA/Grounding)
        │
        └─→ Execute Task
            ├─→ VQA: Generate text response
            └─→ Grounding: Return coordinates


#### 2. *Grounding Pipeline Flow* (Final_Grounding/)

initialize_pipeline()
    │
    ├─→ Load YOLO Model (yolo11n-obb.pt)
    ├─→ Inject LoRA Layers (lora.py)
    ├─→ Load LoRA Weights (lora_weights.pt)
    └─→ Load Qwen3-VL Refiner (refiner.py)

get_coordinates(image_path, query)
    │
    ├─→ YOLO Detection (detector.py)
    │   └─→ Returns: [[cx, cy, w, h, angle], ...]
    │
    └─→ Qwen3-VL Refinement (refiner.py)
        ├─→ Format coordinates as prompt
        ├─→ Process image + query + coordinates
        └─→ Returns: Filtered coordinates matching query


#### 3. *Model Loading Flow* (testing/model_loading.py)

initialize()
    │
    ├─→ Load Qwen3-VL Base Model (quantized 4-bit)
    ├─→ Load Image Classifier (Keras model)
    ├─→ Load Query Classifier
    ├─→ Load SAR LoRA Adapter
    ├─→ Load Optical LoRA Adapter
    └─→ Load FCC Model (if needed)


### Output Formats

#### VQA Output
json
{
    "status": "success",
    "response": "Generated text answer to the visual question"
}


#### Grounding Output
json
{
    "status": "success",
    "response": {
        "yolo_coordinates": [[cx, cy, w, h, angle], ...],
        "final_coordinates": [[cx, cy, w, h, angle], ...],
        "num_detections": 3
    }
}


*Coordinate Format:*
- cx, cy: Normalized center coordinates (0-1)
- w, h: Normalized width and height (0-1)
- angle: Rotation angle in degrees (-90 to 0)

---

## 🔧 Prerequisites

- *Python*: 3.8 or higher
- *CUDA*: 11.8+ (for GPU acceleration)
- *GPU*: NVIDIA GPU with at least 16GB VRAM (recommended)
- *Git LFS*: Required for downloading large model files
- *Operating System*: Linux (recommended), Windows, or macOS

### System Requirements
- *RAM*: 32GB+ recommended
- *Storage*: 50GB+ free space for models
- *CUDA Toolkit*: 12.1+ (if using GPU)

---

## 🚀 Installation & Setup

### Step 1: Clone the Repository

bash
# Clone the repository
git clone <repository-url>
cd model_2041-main

# Initialize Git LFS (required for model files)
git lfs install
git lfs pull


### Step 2: Create Virtual Environment

*On Linux/macOS:*
bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate


*On Windows:*
powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate


### Step 3: Install Dependencies

*For API Server (Recommended):*
bash
# Install API server dependencies
pip install --upgrade pip
pip install -r testing/requirements_api.txt


*For Grounding Module Only:*
bash
# Install grounding module dependencies
pip install --upgrade pip
pip install -r Final_Grounding/requirements.txt


*Note*: The API server requirements include all dependencies needed for the full system.

### Step 4: Verify Installation

bash
# Check Python version
python --version  # Should be 3.8+

# Check CUDA (if using GPU)
python -c "import torch; print(torch.cuda.is_available())"


### Step 5: Download Model Files (if not using Git LFS)

If model files are not automatically downloaded via Git LFS, ensure the following files exist:

- Final_Grounding/yolo11n-obb.pt
- Final_Grounding/lora_weights.pt
- testing/SAR_LORA_ADAPTER/adapter_model.safetensors
- testing/Qwen_for_Optical_2/adapter_model.safetensors
- temp/classifiers/final_model_all_classes.keras

---

## 💻 Usage

### Option 1: Run API Server (Recommended)

The API server provides a RESTful interface for all functionality.

bash
# Navigate to testing directory
cd testing

# Run the API server
python api_server.py


The server will start on http://0.0.0.0:7860

*Example API Request:*
bash
curl -X POST http://localhost:7860/infer \
  -H "Content-Type: application/json" \
  -d '{
    "imageUrl": "path/to/image.jpg",
    "query": "Locate all the airplanes"
  }'


### Option 2: Use Grounding Module Directly

python
from Final_Grounding.main import initialize_pipeline, get_coordinates

# Initialize pipeline (loads models)
initialize_pipeline(
    yolo_path="Final_Grounding/yolo11n-obb.pt",
    lora_path="Final_Grounding/lora_weights.pt"
)

# Get coordinates for an image and query
coords = get_coordinates(
    image_path="path/to/image.jpg",
    query="Locate all the airplanes",
    conf_threshold=0.2
)

print(f"Found {len(coords)} objects:")
for coord in coords:
    print(coord)  # [cx, cy, w, h, angle]


### Option 3: Use Optical Inference Utility

python
from optical import inference_streaming_optimized

# Requires model and tokenizer to be loaded separately
result = inference_streaming_optimized(
    model=model,
    tokenizer=tokenizer,
    image_path="path/to/image.jpg",
    query="What is in this image?",
    max_new_tokens=512
)
print(result)


---

## 🧩 Project Components

### Core Modules

#### 1. *Final_Grounding/* - Object Grounding Pipeline
- *Purpose*: Detect and ground objects in images using YOLO + Qwen3-VL
- *Key Files*:
  - main.py: Simple interface for coordinate extraction
  - src/pipeline.py: Main orchestrator class
  - src/detector.py: YOLO detection with LoRA
  - src/refiner.py: Qwen3-VL coordinate refinement
  - src/lora.py: LoRA layer implementation

#### 2. *testing/* - API Server & Inference Pipeline
- *Purpose*: Production-ready API server with full pipeline
- *Key Files*:
  - api_server.py: Flask REST API server
  - inference.py: Main inference pipeline orchestrator
  - model_loading.py: Model initialization and caching
  - classifier.py: Image type classification
  - query_classifier.py: Query routing
  - vqa_inference.py: Visual Question Answering
  - geo_ground.py: Geographic grounding

#### 3. *temp/* - Development & Training
- *Purpose*: Training notebooks and development files
- *Contents*: Jupyter notebooks for model training and experimentation

### Model Files

All model files are stored in Git LFS. Ensure Git LFS is installed and initialized.

- *YOLO Model*: Final_Grounding/yolo11n-obb.pt
- *LoRA Weights*: Final_Grounding/lora_weights.pt
- *SAR Adapter*: testing/SAR_LORA_ADAPTER/
- *Optical Adapter*: testing/Qwen_for_Optical_2/
- *Image Classifier*: temp/classifiers/final_model_all_classes.keras

---

## 🐳 Docker Deployment

### Build Docker Image

bash
# Build the Docker image
docker build -t model-2041:latest .


### Run Docker Container

bash
# Run with GPU support (NVIDIA Docker required)
docker run --gpus all -p 7860:7860 model-2041:latest

# Or without GPU (CPU only, slower)
docker run -p 7860:7860 model-2041:latest


### Docker Compose (Optional)

Create a docker-compose.yml:

yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "7860:7860"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0,1


Run with:
bash
docker-compose up -d


---

## 🌐 API Endpoints

### Health Check
http
GET /health

*Response:*
json
{"status": "ok"}


### Main Inference Endpoint
http
POST /infer
Content-Type: application/json

{
  "imageUrl": "path/to/image.jpg" or "https://example.com/image.jpg",
  "query": "Your question or grounding query"
}


*Response:*
json
{
  "status": "success",
  "response": "Result based on query type"
}


### Debug Endpoints

#### Query Classification
http
POST /classify_query
Content-Type: application/json

{
  "query": "What is in this image?"
}


#### Image Classification
http
POST /classify_image
Content-Type: application/json

{
  "imageUrl": "path/to/image.jpg"
}


---

## 📊 Model Files

### Required Model Files

| File | Location | Size (approx) | Description |
|------|----------|---------------|-------------|
| yolo11n-obb.pt | Final_Grounding/ | ~6MB | YOLO11n Oriented Bounding Box model |
| lora_weights.pt | Final_Grounding/ | ~50MB | LoRA weights for YOLO |
| adapter_model.safetensors | testing/SAR_LORA_ADAPTER/ | ~500MB | SAR-specific LoRA adapter |
| adapter_model.safetensors | testing/Qwen_for_Optical_2/ | ~500MB | Optical-specific LoRA adapter |
| final_model_all_classes.keras | temp/classifiers/ | ~50MB | Image classifier model |

### Model Downloads

Models are automatically downloaded via Git LFS. If manual download is needed:

1. *Qwen3-VL Base Model*: Automatically downloaded from HuggingFace on first use
   - Model: unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit
   - Requires HuggingFace authentication (if model is gated)

2. *YOLO Model*: Included in repository via Git LFS

3. *LoRA Adapters*: Included in repository via Git LFS

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Git LFS Files Not Downloaded
bash
# Reinstall Git LFS
git lfs install
git lfs pull


#### 2. CUDA Out of Memory
- Reduce batch size in model loading
- Use CPU mode (slower but works)
- Close other GPU applications

#### 3. Model Loading Errors
- Ensure all model files are present
- Check file permissions
- Verify CUDA compatibility

#### 4. Import Errors
bash
# Reinstall dependencies
pip install --upgrade -r testing/requirements_api.txt


#### 5. Port Already in Use
bash
# Change port in api_server.py
app.run(host='0.0.0.0', port=8080)  # Use different port


### GPU Configuration

To specify GPU devices, modify testing/api_server.py:

python
pipeline = InferencePipeline(qwen_gpu=0, geoground_gpu=1)


### Memory Optimization

For systems with limited memory:
- Use CPU mode (set device to "cpu")
- Reduce model quantization (if supported)
- Process images sequentially

---

## 📝 Additional Information

### Environment Variables

Optional environment variables:

bash
# HuggingFace token (if models are gated)
export HF_TOKEN="your_token_here"

# CUDA device selection
export CUDA_VISIBLE_DEVICES=0,1

# Model cache directory
export TRANSFORMERS_CACHE="/path/to/cache"


### Performance Tips

1. *First Request*: Slow (models loading)
2. *Subsequent Requests*: Fast (models cached in memory)
3. *GPU Acceleration*: Significantly faster than CPU
4. *Batch Processing*: Not currently supported (process sequentially)

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff, .tif)
- Other formats supported by PIL/Pillow

### Query Types

*VQA Queries:*
- "What is in this image?"
- "Describe the scene"
- "How many objects are there?"

*Grounding Queries:*
- "Locate all the airplanes"
- "Find the largest building"
- "Where are the ships?"

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

### Development Setup

For development, install additional tools:

bash
pip install pytest black flake8 mypy


---

## 📄 License

[Specify your license here]

---

## 🙏 Acknowledgments

- *YOLO*: Ultralytics YOLO11
- *Qwen3-VL*: Alibaba Cloud Qwen Team
- *Unsloth*: For optimized model loading
- *HuggingFace*: For model hosting and transformers library

---

## 📧 Contact & Support

For issues, questions, or contributions, please open an issue on the repository.

---

*Last Updated*: 2025
