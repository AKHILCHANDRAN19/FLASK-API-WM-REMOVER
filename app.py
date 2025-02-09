from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import requests
import io
import uuid
from skimage.feature import match_template
import base64

app = Flask(__name__)

# Global storage for processed images
PROCESSED_IMAGES = {}

# Watermark URL
WATERMARK_URL = "https://drive.google.com/file/d/1C4yUuDhfQsGe43hO3qmoYLKUpz-t7CUq/view?usp=drivesdk"

def download_watermark():
    """Download watermark image from Google Drive."""
    try:
        file_id = WATERMARK_URL.split('/')[5]
        direct_url = f"https://drive.google.com/uc?id={file_id}"
        response = requests.get(direct_url)
        if response.status_code == 200:
            image_array = np.frombuffer(response.content, np.uint8)
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return None
    except Exception as e:
        print(f"Error downloading watermark: {str(e)}")
        return None

WATERMARK_IMAGE = download_watermark()

def remove_watermark(image, text="Meta AI"):
    """Process image for watermark removal."""
    try:
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create a mask for the bottom right corner
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        h, w = gray.shape
        roi_y = int(h * 0.7)
        roi_x = int(w * 0.7)
        
        # Create text watermark template
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        template = np.zeros((text_size[1] + 10, text_size[0] + 10), dtype=np.uint8)
        cv2.putText(template, text, (5, text_size[1] + 5), font, font_scale, 255, thickness)
        
        # Detect and remove text watermark
        roi = gray[roi_y:, roi_x:]
        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > 0.5:  # Threshold for detection
            top_left = (roi_x + max_loc[0], roi_y + max_loc[1])
            bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
            cv2.rectangle(mask, top_left, bottom_right, 255, -1)
        
        # Remove watermark using inpainting
        processed = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return processed, True
    except Exception as e:
        return None, str(e)

@app.route('/')
def home():
    """Root endpoint showing API documentation."""
    return jsonify({
        "status": "online",
        "message": "Watermark Removal API",
        "endpoints": {
            "/": {
                "method": "GET",
                "description": "This documentation"
            },
            "/api/status": {
                "method": "GET",
                "description": "Check API status"
            },
            "/api/remove-watermark": {
                "method": "POST",
                "description": "Remove watermark from image",
                "parameters": {
                    "image": "file (required)",
                    "text": "string (optional, default: 'Meta AI')"
                },
                "returns": "JSON with processed image info and download URL"
            },
            "/api/download/<image_id>": {
                "method": "GET",
                "description": "Download processed image",
                "returns": "Processed image file"
            }
        }
    })

@app.route('/api/status')
def status():
    """Status endpoint."""
    return jsonify({
        "status": "operational",
        "watermark_loaded": WATERMARK_IMAGE is not None,
        "processed_images": len(PROCESSED_IMAGES)
    })

@app.route('/api/remove-watermark', methods=['POST', 'GET'])
def api_remove_watermark():
    """Watermark removal endpoint."""
    if request.method == 'GET':
        return jsonify({
            "error": "Method not allowed",
            "message": "Please use POST method with an image file",
            "example": {
                "curl": "curl -X POST -F 'image=@your_image.jpg' http://localhost:5000/api/remove-watermark",
                "python": """
                    import requests
                    files = {'image': open('your_image.jpg', 'rb')}
                    response = requests.post('http://localhost:5000/api/remove-watermark', files=files)
                """
            }
        }), 405

    if 'image' not in request.files:
        return jsonify({
            "error": "No image provided",
            "message": "Please provide an image file",
            "status": "error"
        }), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({
            "error": "No selected file",
            "message": "Please select an image file",
            "status": "error"
        }), 400

    try:
        # Read image
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                "error": "Invalid image",
                "message": "Could not process the uploaded file as an image",
                "status": "error"
            }), 400

        # Get watermark text
        watermark_text = request.form.get('text', 'Meta AI')

        # Process image
        processed_image, result = remove_watermark(image, watermark_text)

        if processed_image is None:
            return jsonify({
                "error": "Processing failed",
                "message": str(result),
                "status": "error"
            }), 500

        # Generate ID and save image
        image_id = str(uuid.uuid4())
        success, buffer = cv2.imencode('.png', processed_image)
        if not success:
            return jsonify({
                "error": "Encoding failed",
                "message": "Failed to encode processed image",
                "status": "error"
            }), 500

        PROCESSED_IMAGES[image_id] = buffer.tobytes()

        # Create preview
        preview_size = (400, int(400 * processed_image.shape[0] / processed_image.shape[1]))
        preview = cv2.resize(processed_image, preview_size)
        _, preview_buffer = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
        preview_base64 = base64.b64encode(preview_buffer).decode('utf-8')

        return jsonify({
            "status": "success",
            "message": "Image processed successfully",
            "image_id": image_id,
            "download_url": f"/api/download/{image_id}",
            "preview": f"data:image/jpeg;base64,{preview_base64}"
        })

    except Exception as e:
        return jsonify({
            "error": "Server error",
            "message": str(e),
            "status": "error"
        }), 500

@app.route('/api/download/<image_id>')
def download_image(image_id):
    """Download processed image endpoint."""
    if image_id not in PROCESSED_IMAGES:
        return jsonify({
            "error": "Image not found",
            "message": "The requested image ID does not exist",
            "status": "error"
        }), 404

    return send_file(
        io.BytesIO(PROCESSED_IMAGES[image_id]),
        mimetype='image/png',
        as_attachment=True,
        download_name=f'processed_{image_id}.png'
    )

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({
        "error": "Not found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": ["/", "/api/status", "/api/remove-watermark", "/api/download/<image_id>"]
    }), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return jsonify({
        "error": "Server error",
        "message": str(e),
        "status": "error"
    }), 500

if __name__ == '__main__':
    print("Starting Watermark Removal API...")
    print(f"Watermark image loaded: {WATERMARK_IMAGE is not None}")
    app.run(debug=True)
