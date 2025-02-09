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
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }
        
        .feature-card {
            background: #f8fafc;
            padding: 1.5rem;
            border-radius: 0.5rem;
            text-align: center;
        }
        
        .feature-card i {
            font-size: 2rem;
            color: var(--primary-color);
            margin-bottom: 1rem;
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
        
        <div class="features-grid">
            <div class="feature-card">
                <i class="fas fa-bolt"></i>
                <h3>Fast Processing</h3>
                <p>Advanced algorithms for quick watermark removal</p>
            </div>
            <div class="feature-card">
                <i class="fas fa-image"></i>
                <h3>Batch Processing</h3>
                <p>Process multiple images at once</p>
            </div>
            <div class="feature-card">
                <i class="fas fa-check-circle"></i>
                <h3>High Quality</h3>
                <p>Maintain image quality after processing</p>
            </div>
        </div>
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
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: white;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            box-shadow: 0 2px 4px rgb(0 0 0 / 0.1);
        }
        
        .stat-card i {
            font-size: 1.5rem;
            color: var(--primary-color);
            margin-bottom: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1><i class="fas fa-check-circle"></i> Processing Results</h1>
        
        <div class="stats">
            <div class="stat-card">
                <i class="fas fa-images"></i>
                <h3>{{ images|length }}</h3>
                <p>Images Processed</p>
            </div>
            <div class="stat-card">
                <i class="fas fa-magic"></i>
                <h3>100%</h3>
                <p>Success Rate</p>
            </div>
        </div>
        
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

def read_image_from_file(file_storage):
    """Read an image from a Werkzeug FileStorage and return a cv2 image (BGR)."""
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    file_storage.stream.seek(0)
    return image

def generate_text_template(text, font_scale=1.0, thickness=2):
    """Generate a grayscale template image with the given text."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # Create a black image with some padding
    template = np.zeros((text_height + baseline + 10, text_width + 10), dtype=np.uint8)
    cv2.putText(template, text, (5, text_height + 5), font, font_scale, 255, thickness, cv2.LINE_AA)
    return template

def cv2_to_base64(image):
    """Encode a cv2 image (BGR) as a PNG image in base64."""
    success, buffer = cv2.imencode('.png', image)
    if success:
        return base64.b64encode(buffer).decode('utf-8')
    return None

def multi_scale_match(roi, template, scales):
    """
    Perform multi-scale matching for a given template in the ROI.
    Both roi and template are expected to be preprocessed (grayscale, equalized).
    Returns best score, best location (in ROI coordinates), and the best template shape.
    """
    best_score = -1
    best_loc = None
    best_shape = None
    for s in scales:
        # Resize template
        new_w = int(template.shape[1] * s)
        new_h = int(template.shape[0] * s)
        if new_w < 10 or new_h < 10 or new_w > roi.shape[1] or new_h > roi.shape[0]:
            continue
        resized = cv2.resize(template, (new_w, new_h))
        result = match_template(roi.astype(np.float32), resized.astype(np.float32))
        max_score = result.max()
        if max_score > best_score:
            best_score = max_score
            best_loc = np.unravel_index(np.argmax(result), result.shape)
            best_shape = resized.shape
    return best_score, best_loc, best_shape

def remove_watermark(image, wm_text=None, wm_img=None, threshold=0.5):
    """
    Improved watermark detection and removal.
    Searches for watermark (text and/or image) in the bottom-right region.
    Uses histogram equalization and multi-scale template matching.
    Returns the processed image and detection messages.
    """
    detection_messages = []
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray_image.shape

    # Define ROI in bottom-right (from 70% height and width to the end)
    roi_y = int(h * 0.7)
    roi_x = int(w * 0.7)
    roi = gray_image[roi_y:, roi_x:]
    # Preprocess ROI: histogram equalization
    roi_eq = cv2.equalizeHist(roi)

    scales = np.linspace(0.8, 1.2, 5)  # multi-scale factors

    # Text watermark detection
    if wm_text:
        text_template = generate_text_template(wm_text, font_scale=1.0, thickness=2)
        # Preprocess template: equalize histogram
        text_template_eq = cv2.equalizeHist(text_template)
        best_score, best_loc, best_shape = multi_scale_match(roi_eq, text_template_eq, scales)
        if best_score >= threshold and best_loc is not None:
            abs_y = roi_y + best_loc[0]
            abs_x = roi_x + best_loc[1]
            detection_messages.append(f"Text watermark '{wm_text}' detected at (x={abs_x}, y={abs_y}) with score {best_score:.2f}")
            cv2.rectangle(mask, (abs_x, abs_y), (abs_x + best_shape[1], abs_y + best_shape[0]), 255, -1)
        else:
            detection_messages.append(f"Text watermark '{wm_text}' not confidently detected (max score {best_score:.2f}).")
    
    # Image watermark detection
    if wm_img is not None:
        # Convert watermark image to grayscale and equalize
        if len(wm_img.shape) == 3:
            wm_gray = cv2.cvtColor(wm_img, cv2.COLOR_BGR2GRAY)
        else:
            wm_gray = wm_img
        wm_eq = cv2.equalizeHist(wm_gray)
        best_score_img, best_loc_img, best_shape_img = multi_scale_match(roi_eq, wm_eq, scales)
        if best_score_img >= threshold and best_loc_img is not None:
            abs_y_img = roi_y + best_loc_img[0]
            abs_x_img = roi_x + best_loc_img[1]
            detection_messages.append(f"Image watermark detected at (x={abs_x_img}, y={abs_y_img}) with score {best_score_img:.2f}")
            cv2.rectangle(mask, (abs_x_img, abs_y_img), (abs_x_img + best_shape_img[1], abs_y_img + best_shape_img[0]), 255, -1)
        else:
            detection_messages.append(f"Image watermark not confidently detected (max score {best_score_img:.2f}).")
    
    # Inpaint the detected regions
    processed = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return processed, detection_messages

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("images")
        wm_text = request.form.get("watermark_text", "").strip() or "Meta AI"
        
        results = []
        for file in files:
            if file and file.filename != "":
                try:
                    filename = secure_filename(file.filename)
                    orig_image = read_image_from_file(file)
                    if orig_image is None:
                        continue
                    
                    processed_image, detection_msgs = remove_watermark(orig_image, wm_text, DEFAULT_WATERMARK, threshold=0.5)
                    orig_b64 = cv2_to_base64(orig_image)
                    proc_b64 = cv2_to_base64(processed_image)
                    
                    if orig_b64 and proc_b64:
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
                except Exception as e:
                    print(f"Error processing {file.filename}: {str(e)}")
                    continue
        
        return render_template_string(RESULT_HTML, images=results)
    return render_template_string(INDEX_HTML)

@app.route("/download/<uid>")
def download(uid):
    if uid in PROCESSED_IMAGES:
        return send_file(
            io.BytesIO(PROCESSED_IMAGES[uid]),
            mimetype='image/png',
            as_attachment=True,
            download_name=f"processed_{uid}.png"
        )
    return "File not found", 404

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
