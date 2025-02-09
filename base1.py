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
        }
        
        @media (max-width: 768px) {
            .container {
                width: 95%;
                padding: 1rem;
                margin: 1rem auto;
            }
        }
        
        h1 {
            color: var(--primary-color);
            margin-bottom: 2rem;
            text-align: center;
            font-size: clamp(1.8rem, 4vw, 2.5rem);
            text-shadow: 0 0 15px rgba(109, 40, 217, 0.5);
        }
        
        .upload-section {
            background: rgba(255, 255, 255, 0.05);
            padding: 2rem;
            border-radius: 1rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        @media (max-width: 480px) {
            .upload-section {
                padding: 1rem;
            }
        }
        
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--text-primary);
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
        }
        
        input[type="file"]:hover,
        input[type="text"]:hover {
            border-color: var(--primary-color);
        }
        
        button {
            background: var(--primary-color);
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
        }
        
        button:hover {
            background: var(--primary-hover);
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
        
        .info-text {
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }
        
        @media (max-width: 768px) {
            .features-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 1.5rem;
            border-radius: 1rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(5px);
        }
        
        .feature-card i {
            font-size: 2rem;
            color: var(--primary-color);
            margin-bottom: 1rem;
            text-shadow: 0 0 10px rgba(109, 40, 217, 0.5);
        }
        
        .feature-card h3 {
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }
        
        .feature-card p {
            color: var(--text-secondary);
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
            --accent: #7c3aed;
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
            padding: 1rem;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: auto;
            padding: 1rem;
        }
        
        h1 {
            color: var(--primary-color);
            margin-bottom: 2rem;
            text-align: center;
            font-size: clamp(1.8rem, 4vw, 2.5rem);
            text-shadow: 0 0 15px rgba(109, 40, 217, 0.5);
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: var(--bg-card);
            padding: 1.5rem;
            border-radius: 1rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(5px);
        }
        
        .stat-card i {
            font-size: 1.5rem;
            color: var(--primary-color);
            margin-bottom: 0.5rem;
        }
        
        .image-pair {
            background: var(--bg-card);
            padding: 2rem;
            border-radius: 1rem;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            animation: fadeIn 0.5s ease-out;
        }
        
        @media (max-width: 768px) {
            .image-pair {
                padding: 1rem;
            }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .image-pair h3 {
            color: var(--primary-color);
            margin-bottom: 1.5rem;
            font-size: clamp(1.2rem, 3vw, 1.5rem);
        }
        
        .image-comparison {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 2rem;
            margin-bottom: 1.5rem;
        }
        
        .image-container {
            position: relative;
            overflow: hidden;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .image-container img {
            width: 100%;
            height: auto;
            display: block;
            border-radius: 0.5rem;
        }
        
        .image-label {
            position: absolute;
            top: 1rem;
            left: 1rem;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            font-size: 0.875rem;
            backdrop-filter: blur(5px);
        }
        
        .detection-info {
            background: rgba(255, 255, 255, 0.05);
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1.5rem 0;
            font-size: 0.875rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .detection-info strong {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
            color: var(--primary-color);
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
            box-shadow: 0 4px 15px rgba(109, 40, 217, 0.3);
        }
        
        .download-btn:hover {
            background: var(--primary-hover);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(109, 40, 217, 0.4);
        }
        
        
