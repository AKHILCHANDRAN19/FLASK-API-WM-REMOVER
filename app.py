import cv2
import numpy as np
import base64
import io
import uuid
from flask import Flask, request, render_template_string, send_file
from werkzeug.utils import secure_filename
from skimage.feature import match_template

app = Flask(__name__)

# Global dictionary to hold processed image bytes (for download)
PROCESSED_IMAGES = {}

# HTML templates (embedded HTML, CSS, and minimal JS)
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Watermark Remover</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: auto; }
        label { font-weight: bold; }
        input, button { margin-top: 5px; }
    </style>
</head>
<body>
<div class="container">
    <h1>Watermark Remover</h1>
    <form method="POST" enctype="multipart/form-data">
        <label>Select images (multiple allowed):</label><br>
        <input type="file" name="images" multiple required><br><br>
        <label>Watermark Text (default is "Meta AI"):</label><br>
        <input type="text" name="watermark_text" placeholder="Enter watermark text"><br><br>
        <label>Upload Watermark Image (optional):</label><br>
        <input type="file" name="watermark_image"><br><br>
        <button type="submit">Remove Watermark</button>
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
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1000px; margin: auto; }
        .image-pair { margin-bottom: 30px; padding-bottom: 10px; border-bottom: 1px solid #ccc; }
        .image-pair img { max-width: 300px; margin-right: 10px; border: 1px solid #ddd; }
        .download { margin-top: 10px; }
        .detection { background: #f0f0f0; padding: 5px; margin-top: 5px; font-size: 0.9em; }
    </style>
</head>
<body>
<div class="container">
    <h1>Results</h1>
    {% for item in images %}
    <div class="image-pair">
        <h3>{{ item.filename }}</h3>
        <div>
            <strong>Before:</strong><br>
            <img src="data:image/png;base64,{{ item.original }}" alt="Original Image">
        </div>
        <div>
            <strong>After:</strong><br>
            <img src="data:image/png;base64,{{ item.processed }}" alt="Processed Image">
        </div>
        <div class="detection">
            <strong>Detection Info:</strong><br>
            {% for msg in item.detections %}
                {{ msg }}<br>
            {% endfor %}
        </div>
        <div class="download">
            <a href="/download/{{ item.uid }}">Download Processed Image</a>
        </div>
    </div>
    {% endfor %}
    <br><a href="/">&#8592; Back to Upload</a>
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
        # Use watermark text if provided; default to "Meta AI" if empty
        wm_text = request.form.get("watermark_text", "").strip() or "Meta AI"
        watermark_image_file = request.files.get("watermark_image")
        wm_img = None
        if watermark_image_file and watermark_image_file.filename != "":
            watermark_image_file.filename = secure_filename(watermark_image_file.filename)
            wm_img = read_image_from_file(watermark_image_file)
        
        results = []
        # Process each uploaded image
        for file in files:
            if file and file.filename != "":
                filename = secure_filename(file.filename)
                orig_image = read_image_from_file(file)
                processed_image, detection_msgs = remove_watermark(orig_image, wm_text, wm_img, threshold=0.5)
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
    app.run(debug=True)
