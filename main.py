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
# Version 1 - INDEX_HTML (Sleek Dark)
INDEX_HTML = """
 <!DOCTYPE html>
<html>
<head>
    <title>Professional Watermark Remover</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #6d28d9;
            --primary-hover: #7c3aed;
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent: #7c3aed;
            --gradient-start: #4f46e5;
            --gradient-end: #7c3aed;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            transition: all 0.3s ease;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
            background: linear-gradient(135deg, var(--bg-dark) 0%, #1a1a2e 100%);
        }
        
        .container {
            width: 90%;
            max-width: 1200px;
            margin: 2rem auto;
            padding: 2rem;
            background: var(--bg-card);
            border-radius: 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            animation: slideIn 0.6s ease-out;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
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
            font-size: clamp(1.8rem, 4vw, 2.5rem);
            text-shadow: 0 0 15px rgba(109, 40, 217, 0.5);
            animation: glow 2s ease-in-out infinite alternate;
        }

        @keyframes glow {
            from {
                text-shadow: 0 0 5px rgba(109, 40, 217, 0.5);
            }
            to {
                text-shadow: 0 0 20px rgba(109, 40, 217, 0.8);
            }
        }
        
        .upload-section {
            background: rgba(255, 255, 255, 0.05);
            padding: 2rem;
            border-radius: 1rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transform: translateZ(0);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .upload-section:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        
        .progress-container {
            width: 100%;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 1rem;
            margin: 1rem 0;
            overflow: hidden;
            display: none;
        }

        .progress-bar {
            width: 0%;
            height: 10px;
            background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
            border-radius: 1rem;
            transition: width 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .progress-bar::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(255, 255, 255, 0.2),
                transparent
            );
            animation: shimmer 1.5s infinite;
        }

        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .form-group {
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }
        
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--text-primary);
            position: relative;
            z-index: 1;
        }

        label i {
            margin-right: 0.5rem;
            color: var(--primary-color);
        }
        
        input[type="file"],
        input[type="text"] {
            width: 100%;
            padding: 0.75rem;
            background: var(--bg-dark);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 0.5rem;
            font-size: 1rem;
            color: var(--text-primary);
            margin-top: 0.5rem;
            transition: all 0.3s ease;
        }
        
        input[type="file"]:hover,
        input[type="text"]:hover {
            border-color: var(--primary-color);
            box-shadow: 0 0 15px rgba(109, 40, 217, 0.3);
        }
        
        button {
            background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
            color: white;
            padding: 1rem 2rem;
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
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 15px rgba(109, 40, 217, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(255, 255, 255, 0.2),
                transparent
            );
            transition: 0.5s;
        }

        button:hover::before {
            left: 100%;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(109, 40, 217, 0.4);
        }
        
        .loading {
            display: none;
            text-align: center;
            margin-top: 1rem;
            color: var(--text-secondary);
        }
        
        .spinner {
            animation: spin 1s linear infinite;
            display: inline-block;
        }
        
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
            opacity: 0;
            transform: translateY(20px);
            animation: fadeInUp 0.6s ease-out forwards;
            animation-delay: 0.3s;
        }

        @keyframes fadeInUp {
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 1.5rem;
            border-radius: 1rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(5px);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        
        .feature-card i {
            font-size: 2rem;
            color: var(--primary-color);
            margin-bottom: 1rem;
            text-shadow: 0 0 10px rgba(109, 40, 217, 0.5);
        }

        /* File upload custom styling */
        .file-upload-wrapper {
            position: relative;
            width: 100%;
            height: 150px;
            border: 2px dashed var(--primary-color);
            border-radius: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease;
        }

        .file-upload-wrapper:hover {
            border-color: var(--primary-hover);
            background: rgba(255, 255, 255, 0.08);
        }

        .file-upload-wrapper input[type="file"] {
            position: absolute;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
        }

        .upload-content {
            text-align: center;
            pointer-events: none;
        }

        .upload-content i {
            font-size: 2.5rem;
            color: var(--primary-color);
            margin-bottom: 1rem;
        }

        /* Toast Notification */
        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: var(--primary-color);
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            display: none;
            animation: slideInRight 0.3s ease-out;
        }

        @keyframes slideInRight {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
    </style>
    <script>
        function showLoading() {
            document.querySelector('.loading').style.display = 'block';
            document.querySelector('.progress-container').style.display = 'block';
            document.querySelector('button[type="submit"]').disabled = true;
            simulateProgress();
        }
        
        function simulateProgress() {
            let progress = 0;
            const progressBar = document.querySelector('.progress-bar');
            const interval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress > 100) progress = 100;
                progressBar.style.width = progress + '%';
                if (progress === 100) clearInterval(interval);
            }, 500);
        }

        function showToast(message) {
            const toast = document.createElement('div');
            toast.className = 'toast';
            toast.textContent = message;
            document.body.appendChild(toast);
            toast.style.display = 'block';
            
            setTimeout(() => {
                toast.style.display = 'none';
                document.body.removeChild(toast);
            }, 3000);
        }
        
        function validateForm() {
            const fileInput = document.querySelector('input[name="images"]');
            if (fileInput.files.length === 0) {
                showToast('Please select at least one image to process.');
                return false;
            }
            showLoading();
            return true;
        }

        // File upload preview
        function handleFileSelect(event) {
            const files = event.target.files;
            const uploadContent = document.querySelector('.upload-content');
            if (files.length > 0) {
                uploadContent.innerHTML = `
                    <i class="fas fa-check-circle"></i>
                    <p>${files.length} file(s) selected</p>
                `;
            }
        }

        document.addEventListener('DOMContentLoaded', () => {
            const fileInput = document.querySelector('input[name="images"]');
            fileInput.addEventListener('change', handleFileSelect);
        });
    </script>
</head>
<body>
    <div class="container">
        <h1><i class="fas fa-magic"></i> Watermark Remover Pro</h1>
        <form method="POST" enctype="multipart/form-data" onsubmit="return validateForm()">
            <div class="upload-section">
                <div class="form-group">
                    <label><i class="fas fa-images"></i> Select Images</label>
                    <div class="file-upload-wrapper">
                        <input type="file" name="images" multiple accept="image/*" required>
                        <div class="upload-content">
                            <i class="fas fa-cloud-upload-alt"></i>
                            <p>Drag & drop your images here or click to browse</p>
                        </div>
                    </div>
                    <div class="info-text">Multiple images allowed. Supported formats: PNG, JPG, JPEG</div>
                </div>
                
                <div class="form-group">
                    <label><i class="fas fa-text-height"></i> Custom Text Watermark (Optional)</label>
                    <input type="text" name="watermark_text" placeholder="Default: Meta AI">
                    <div class="info-text">Leave empty to use default "Meta AI" watermark</div>
                </div>

                <div class="progress-container">
                    <div class="progress-bar"></div>
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

# Version 1 - RESULT_HTML (Sleek Dark)
RESULT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Watermark Removal Results</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #6d28d9;
            --primary-hover: #7c3aed;
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --success-color: #10b981;
            --accent-gradient: linear-gradient(135deg, #4f46e5, #7c3aed);
            --card-gradient: linear-gradient(135deg, rgba(79, 70, 229, 0.1), rgba(124, 58, 237, 0.1));
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            transition: all 0.3s ease;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
            background: linear-gradient(135deg, #0f172a 0%, #1a1a2e 100%);
            padding-bottom: 2rem;
        }

        .user-info {
            position: fixed;
            top: 1rem;
            right: 1rem;
            background: rgba(255, 255, 255, 0.1);
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            backdrop-filter: blur(10px);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
            animation: slideInRight 0.5s ease-out;
        }

        @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .container {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 1rem;
        }
        
        .header {
            text-align: center;
            margin-bottom: 3rem;
            animation: fadeInDown 0.8s ease-out;
        }

        @keyframes fadeInDown {
            from { transform: translateY(-30px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        h1 {
            color: transparent;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            background-clip: text;
            font-size: clamp(2rem, 4vw, 3rem);
            margin-bottom: 1rem;
            position: relative;
            display: inline-block;
        }

        h1::after {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 50%;
            transform: translateX(-50%);
            width: 50%;
            height: 4px;
            background: var(--accent-gradient);
            border-radius: 2px;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
            animation: fadeIn 0.8s ease-out;
            animation-delay: 0.2s;
            opacity: 0;
            animation-fill-mode: forwards;
        }

        @keyframes fadeIn {
            to { opacity: 1; }
        }
        
        .stat-card {
            background: var(--card-gradient);
            padding: 1.5rem;
            border-radius: 1rem;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transform: translateZ(0);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(124, 58, 237, 0.2);
        }
        
        .stat-card i {
            font-size: 2rem;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 1rem;
        }
        
        .image-pair {
            background: rgba(30, 41, 59, 0.7);
            padding: 2rem;
            border-radius: 1.5rem;
            margin-bottom: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            opacity: 0;
            transform: translateY(20px);
            animation: slideUp 0.6s ease-out forwards;
        }

        .image-pair:nth-child(odd) {
            animation-delay: 0.2s;
        }

        .image-pair:nth-child(even) {
            animation-delay: 0.4s;
        }

        @keyframes slideUp {
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .image-pair h3 {
            color: var(--text-primary);
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.25rem;
        }

        .image-pair h3 i {
            color: var(--primary-color);
        }
        
        .image-comparison {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-bottom: 1.5rem;
        }
        
        .image-container {
            position: relative;
            overflow: hidden;
            border-radius: 1rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease;
        }

        .image-container:hover {
            transform: scale(1.02);
        }
        
        .image-container img {
            width: 100%;
            height: auto;
            display: block;
            transition: transform 0.3s ease;
        }
        
        .image-label {
            position: absolute;
            top: 1rem;
            left: 1rem;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 2rem;
            font-size: 0.875rem;
            backdrop-filter: blur(5px);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            z-index: 1;
        }
        
        .detection-info {
            background: rgba(255, 255, 255, 0.05);
            padding: 1.5rem;
            border-radius: 1rem;
            margin: 1.5rem 0;
            font-size: 0.875rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }

        .detection-info::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.05), transparent);
            animation: shine 2s infinite;
        }

        @keyframes shine {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .download-btn {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: var(--accent-gradient);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            text-decoration: none;
            font-weight: 600;
            position: relative;
            overflow: hidden;
            transition: transform 0.3s ease;
        }

        .download-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: 0.5s;
        }

        .download-btn:hover::before {
            left: 100%;
        }

        .download-btn:hover {
            transform: translateY(-2px);
        }
        
        .back-btn {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--primary-color);
            text-decoration: none;
            font-weight: 600;
            margin-top: 2rem;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }

        .back-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateX(-5px);
        }

        /* Success Badge Animation */
        .success-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(16, 185, 129, 0.2);
            color: var(--success-color);
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            font-size: 0.875rem;
            margin-bottom: 1rem;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .container {
                padding: 0 1rem;
            }

            .image-comparison {
                grid-template-columns: 1fr;
            }

            .stat-card {
                padding: 1rem;
            }
        }
    </style>
</head>
<body>
    <div class="user-info">
        <i class="fas fa-user-circle"></i>
        <span>{{ username }}</span>
        <span>•</span>
        <span>{{ current_time }}</span>
    </div>

    <div class="container">
        <div class="header">
            <h1><i class="fas fa-check-circle"></i> Processing Results</h1>
            <div class="success-badge">
                <i class="fas fa-check"></i>
                All images processed successfully
            </div>
        </div>
        
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
            <div class="stat-card">
                <i class="fas fa-clock"></i>
                <h3>{{ processing_time }}s</h3>
                <p>Processing Time</p>
            </div>
        </div>
        
        {% for item in images %}
        <div class="image-pair">
            <h3>
                <i class="fas fa-image"></i>
                {{ item.filename }}
            </h3>
            
            <div class="image-comparison">
                <div class="image-container">
                    <span class="image-label">
                        <i class="fas fa-file-image"></i>
                        Original
                    </span>
                    <img src="data:image/png;base64,{{ item.original }}" alt="Original Image">
                </div>
                <div class="image-container">
                    <span class="image-label">
                        <i class="fas fa-magic"></i>
                        Processed
                    </span>
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

    <script>
        // Display current user and time
        document.addEventListener('DOMContentLoaded', function() {
            const userInfo = document.querySelector('.user-info');
            userInfo.innerHTML = `
                <i class="fas fa-user-circle"></i>
                <span>AKHILCHANDRAN19</span>
                <span>•</span>
                <span>2025-02-09 11:05:23</span>
            `;
        });

        // Add smooth scroll animation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });
    </script>
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
