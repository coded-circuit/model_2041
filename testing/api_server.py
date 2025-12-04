from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import InferencePipeline
import os
import requests
import uuid

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==========================================
# GLOBAL PIPELINE (LOADS MODELS ONCE!) . .. 
# ==========================================
pipeline = None

def get_pipeline():
    """
    Creates pipeline ONCE and reuses it.
    Models load only on first request, then stay in memory.
    """
    global pipeline
    if pipeline is None:
        print("🔧 Initializing Pipeline (FIRST TIME ONLY)...")
        # Specify GPU assignments here
        pipeline = InferencePipeline(qwen_gpu=0, geoground_gpu=1)
        pipeline.initialize()  # Load models once
        print("✅ Pipeline Ready!")
    return pipeline

get_pipeline()

def _download_if_url(image_url: str) -> str:
    """
    If image_url is an http(s) URL, download it to UPLOAD_FOLDER and
    return the local file path. Otherwise assume it is already a path.
    """
    if image_url.startswith("http://") or image_url.startswith("https://"):
        filename = f"{uuid.uuid4().hex}.jpg"
        local_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resp = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return local_path
    return image_url


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ==========================================
# MAIN ENDPOINT - USES YOUR PIPELINE.RUN()
# ==========================================

@app.route('/infer', methods=['POST'])
def infer():
    """
    Main endpoint that uses your complete pipeline.
    
    Input:  { "imageUrl": "...", "query": "..." }
    Output: { "response": "...", "status": "success" }
    
    This automatically handles:
    - Image classification (SAR/Optical/FCC)
    - Query classification (VQA/Grounding/etc)
    - Memory management
    - Parallel processing
    - Everything in your pipeline!
    """
    try:
        data = request.get_json(silent=True) or {}
        image_url = (data.get('imageUrl') or '').strip()
        user_query = (data.get('query') or '').strip()

        if not image_url:
            return jsonify({'status': 'error', 'message': 'imageUrl required'}), 400
        if not user_query:
            return jsonify({'status': 'error', 'message': 'query required'}), 400

        # Download image if it's a URL
        local_path = _download_if_url(image_url)
        if not os.path.exists(local_path):
            return jsonify({'status': 'error', 'message': 'Image not found'}), 404

        # 🎯 THIS IS THE KEY - Use your main pipeline.run() method
        result = get_pipeline().run(local_path, user_query)
        
        return jsonify({
            'status': 'success',
            'response': result
        }), 200
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ==========================================
# OPTIONAL: DEBUG ENDPOINTS (Keep if needed)
# ==========================================

@app.route('/classify_query', methods=['POST'])
def classify_query_endpoint():
    """
    Debug endpoint to see how a query would be classified.
    """
    try:
        data = request.get_json(silent=True) or {}
        user_query = (data.get('query') or '').strip()
        
        if not user_query:
            return jsonify({'status': 'error', 'message': 'query required'}), 400
        
        from query_classifier import classify_query
        tasks = classify_query(get_pipeline().zoo, user_query)
        
        return jsonify({
            'status': 'success',
            'tasks': tasks
        }), 200
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/classify_image', methods=['POST'])
def classify_image_endpoint():
    """
    Debug endpoint to see image type classification.
    """
    try:
        data = request.get_json(silent=True) or {}
        image_url = (data.get('imageUrl') or '').strip()
        
        if not image_url:
            return jsonify({'status': 'error', 'message': 'imageUrl required'}), 400
        
        local_path = _download_if_url(image_url)
        if not os.path.exists(local_path):
            return jsonify({'status': 'error', 'message': 'Image not found'}), 404
        
        from classifier import classify_image
        img_type = classify_image(get_pipeline().zoo, local_path)
        
        return jsonify({
            'status': 'success',
            'image_type': img_type
        }), 200
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    # Bind to all interfaces so it works inside Docker / on VastAI
    app.run(host='0.0.0.0', port=7860, debug=False)

