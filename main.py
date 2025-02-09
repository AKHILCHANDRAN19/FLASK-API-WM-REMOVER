import cv2
import numpy as np
import base64
import io
import uuid
import requests
from flask import Flask, request, render_template_string, send_file
from werkzeug.utils import secure_filename
from skimage.feature import match_template

app = Flask(__name__)

# Global dictionary to hold processed image bytes (for download)
PROCESSED_IMAGES = {}

# Download and store the default watermark image
DEFAULT_WATERMARK_URL = "https://drive.google.com/uc?export=download&id=1C4yUuDhfQsGe43hO3qmoYLKUpz-t7CUq"
try:
    response = requests.get(DEFAULT_WATERMARK_URL)
    if response.status_code == 200:
        DEFAULT_WATERMARK = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    else:
        DEFAULT_WATERMARK = None
except:
    DEFAULT_WATERMARK = None

# HTML templates with enhanced styling and animations
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Professional Watermark Remover</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #1e40af;
            --bg-color: #f8fafc;
            --text-color: #1e293b;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            transition: all 0.3s ease;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }
        
        .container {
            max-width: 800px;
            margin: 2rem auto;
            padding: 2rem;
            background: white;
            border-radius: 1rem;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            animation: slideUp 0.5s ease-out;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        h1 {
            color: var(--primary-color);
            margin-bottom: 2rem;
            text-align: center;
            font-size: 2.5rem;
        }
        
        .upload-section {
            background: #f8fafc;
            padding: 2rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
        }
        
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--text-color);
        }
        
        input[type="file"],
        input[type="text"] {
            width: 100%;
            padding: 0.75rem;
            border: 2px solid #e2e8f0;
            border-radius: 0.5rem;
            font-size: 1rem;
            margin-top: 0.5rem;
        }
        
        input[type="file"]:hover,
        input[type="text"]:hover {
            border-color: var(--primary-color);
        }
        
        button {
            background: var(--primary-color);
            color: white;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 0.5rem;
            font-size: 1rem;
            cursor: pointer;
            width: 100%;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        button:hover {
            background: var(--secondary-color);
            transform: translateY(-1px);
        }
        
        .loading {
            display: none;
            text-align: center;
            margin-top: 1rem;
        }
        
        .spinner {
            animation: spin 1s linear infinite;
            display: inline-block;
        }
        
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .info-text {
            color: #64748b;
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }
    </style>
    <script>
        function showLoading() {
            document.querySelector('.loading').style.display = 'block';
            document.querySelector('button[type="submit"]').disabled = true;
        }
        
        function validateForm() {
            const fileInput = document.querySelector('input[name="images"]');
            if (fileInput.files.length === 0) {
                alert('Please select at least one image to process.');
                return false;
            }
            showLoading();
            return true;
        }
    </script>
</head>
<body>
    <div class="container">
        <h1><i class="fas fa-magic"></i> Watermark Remover Pro</h1>
        <form method="POST" enctype="multipart/form-data" onsubmit="return validateForm()">
            <div class="upload-section">
                <div class="form-group">
                    <label><i class="fas fa-images"></i> Select Images</label>
                    <input type="file" name="images" multiple accept="image/*" required>
                    <div class="info-text">Multiple images allowed. Supported formats: PNG, JPG, JPEG</div>
                </div>
                
                <div class="form-group">
                    <label><i class="fas fa-text-height"></i> Custom Text Watermark (Optional)</label>
                    <input type="text" name="watermark_text" placeholder="Default: Meta AI">
                    <div class="info-text">Leave empty to use default "Meta AI" watermark</div>
                </div>
            </div>
            
            <button type="submit">
                <i class="fas fa-magic"></i> Remove Watermarks
            </button>
            
            <div class="loading">
                <i class="fas fa-spinner spinner"></i> Processing images...
            </div>
        </form>
    </div>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Watermark Removal Results</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #1e40af;
            --bg-color: #f8fafc;
            --text-color: #1e293b;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            transition: all 0.3s ease;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 2rem;
        }
        
        .container {
            max-width: 1200px;
            margin: auto;
        }
        
        h1 {
            color: var(--primary-color);
            margin-bottom: 2rem;
            text-align: center;
            font-size: 2.5rem;
        }
        
        .image-pair {
            background: white;
            padding: 2rem;
            border-radius: 1rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            animation: fadeIn 0.5s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .image-pair h3 {
            color: var(--primary-color);
            margin-bottom: 1rem;
        }
        
        .image-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 1.5rem;
        }
        
        .image-container {
            position: relative;
        }
        
        .image-container img {
            max-width: 100%;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgb(0 0 0 / 0.1);
        }
        
        .image-label {
            position: absolute;
            top: 1rem;
            left: 1rem;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            font-size: 0.875rem;
        }
        
        .detection-info {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            font-size: 0.875rem;
        }
        
        .download-btn {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: var(--primary-color);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .download-btn:hover {
            background: var(--secondary-color);
            transform: translateY(-1px);
        }
        
        .back-btn {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--primary-color);
            text-decoration: none;
            font-weight: 600;
            margin-top: 2rem;
        }
        
        .back-btn:hover {
            color: var(--secondary-color);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1><i class="fas fa-check-circle"></i> Processing Results</h1>
        
        {% for item in images %}
        <div class="image-pair">
            <h3><i class="fas fa-image"></i> {{ item.filename }}</h3>
            
            <div class="image-comparison">
                <div class="image-container">
                    <span class="image-label">Original</span>
                    <img src="data:image/png;base64,{{ item.original }}" alt="Original Image">
                </div>
                <div class="image-container">
                    <span class="image-label">Processed</span>
                    <img src="data:image/png;base64,{{ item.processed }}" alt="Processed Image">
                </div>
            </div>
            
            <div class="detection-info">
                <strong><i class="fas fa-info-circle"></i> Detection Details:</strong><br>
                {% for msg in item.detections %}
                    {{ msg }}<br>
                {% endfor %}
            </div>
            
            <a href="/download/{{ item.uid }}" class="download-btn">
                <i class="fas fa-download"></i> Download Processed Image
            </a>
        </div>
        {% endfor %}
        
        <a href="/" class="back-btn">
            <i class="fas fa-arrow-left"></i> Process More Images
        </a>
    </div>
</body>
</html>
"""

[Previous code for image processing functions remains the same...]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("images")
        wm_text = request.form.get("watermark_text", "").strip() or "Meta AI"
        
        results = []
        for file in files:
            if file and file.filename != "":
                filename = secure_filename(file.filename)
                orig_image = read_image_from_file(file)
                processed_image, detection_msgs = remove_watermark(orig_image, wm_text, DEFAULT_WATERMARK, threshold=0.5)
                orig_b64 = cv2_to_base64(orig_image)
                proc_b64 = cv2_to_base64(processed_image)
                uid = str(uuid.uuid4())
                success, proc_buffer = cv2.imencode('.png', processed_image)
                if success:
                    PROCESSED_IMAGES[uid] = proc_buffer.tobytes()
                results.append({
                    "uid": uid,
                    "filename": filename,
                    "original": orig_b64,
                    "processed": proc_b64,
                    "detections": detection_msgs
                })
        return render_template_string(RESULT_HTML, images=results)
    return render_template_string(INDEX_HTML)

