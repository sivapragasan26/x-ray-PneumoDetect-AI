<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PneumoDetect AI | Ayushi Rathour - Healthcare Innovation</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🫁</text></svg>">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(300deg,#211c6a,#17594a,#08045b,#264422,#b7b73d);
            background-size: 300% 300%;
            animation: gradient-animation 25s ease infinite;
            min-height: 100vh;
            color: #333;
            overflow-x: hidden;
        }

        @keyframes gradient-animation {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }

        /* Header */
        .header {
            text-align: center;
            padding: 60px 0 40px;
            color: white;
        }

        .logo {
            font-size: 4rem;
            margin-bottom: 1rem;
            filter: drop-shadow(0 0 20px rgba(255,255,255,0.3));
        }

        .main-title {
            font-size: clamp(2.5rem, 5vw, 4rem);
            font-weight: 800;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 10px rgba(0,0,0,0.3);
            letter-spacing: -1px;
        }

        .subtitle {
            font-size: clamp(1rem, 2.5vw, 1.3rem);
            opacity: 0.9;
            margin-bottom: 2rem;
            font-weight: 500;
        }

        .hero-description {
            max-width: 600px;
            margin: 0 auto;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(25px);
            padding: 20px 30px;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.2);
            font-weight: 500;
            font-size: clamp(0.95rem, 2vw, 1.1rem);
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 40px 0;
        }

        /* Modern Glassmorphic Cards */
        .stat-card {
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(25px);
            padding: 32px 24px;
            border-radius: 24px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.12);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            transition: left 0.6s ease;
        }

        .stat-card:hover::before {
            left: 100%;
        }

        .stat-card:hover {
            transform: translateY(-8px) scale(1.02);
            background: rgba(255,255,255,0.12);
            border-color: rgba(255,255,255,0.25);
            box-shadow: 0 20px 60px rgba(0,0,0,0.25);
        }

        .stat-number {
            font-size: 2.8rem;
            font-weight: 800;
            color: white;
            margin-bottom: 12px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .stat-label {
            color: rgba(255,255,255,0.85);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1.2px;
        }

        /* Modern Glassmorphic Upload Section */
        .upload-section {
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(25px);
            border-radius: 28px;
            padding: 48px;
            margin: 48px 0;
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
            transition: all 0.3s ease;
        }

        .upload-section:hover {
            background: rgba(255,255,255,0.1);
            border-color: rgba(255,255,255,0.18);
        }

        .upload-title {
            text-align: center;
            font-size: 1.6rem;
            font-weight: 700;
            color: white;
            margin-bottom: 32px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .upload-area {
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 20px;
            padding: 48px;
            text-align: center;
            background: rgba(255,255,255,0.05);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }

        .upload-area::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 70%);
            transition: opacity 0.3s ease;
            opacity: 0;
        }

        .upload-area:hover::before {
            opacity: 1;
        }

        .upload-area:hover {
            border-color: rgba(255,255,255,0.5);
            background: rgba(255,255,255,0.1);
            transform: translateY(-4px);
            box-shadow: 0 12px 48px rgba(0,0,0,0.15);
        }

        .upload-area.dragover {
            border-color: rgba(255,255,255,0.6);
            background: rgba(255,255,255,0.15);
            transform: scale(1.02);
        }

        .upload-icon {
            font-size: 3.5rem;
            margin-bottom: 16px;
            filter: drop-shadow(0 4px 12px rgba(0,0,0,0.2));
        }

        .upload-text {
            font-size: 1.2rem;
            font-weight: 600;
            color: white;
            margin-bottom: 8px;
            text-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }

        .upload-subtext {
            color: rgba(255,255,255,0.7);
            font-size: 0.9rem;
            font-weight: 500;
        }

        .file-input {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
        }

        /* Image Preview */
        .image-preview {
            margin: 30px 0;
            text-align: center;
        }

        .preview-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        /* Modern Glassmorphic Button */
        .analyze-btn {
            display: block;
            width: 100%;
            max-width: 320px;
            margin: 32px auto;
            padding: 18px 36px;
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(25px);
            color: white;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 60px;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
            position: relative;
            overflow: hidden;
            text-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }

        .analyze-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.6s ease;
        }

        .analyze-btn:hover::before {
            left: 100%;
        }

        .analyze-btn:hover {
            transform: translateY(-4px);
            background: rgba(255,255,255,0.2);
            border-color: rgba(255,255,255,0.3);
            box-shadow: 0 16px 48px rgba(0,0,0,0.25);
        }

        .analyze-btn:active {
            transform: translateY(-2px);
        }

        .analyze-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .analyze-btn:disabled:hover {
            transform: none;
            background: rgba(255,255,255,0.15);
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        }

        /* Results Section */
        .results-section {
            margin: 40px 0;
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            animation: slideIn 0.5s ease;
        }

        .results-pneumonia {
            background: linear-gradient(135deg, #ff6b6b, #ff5252);
            color: white;
            box-shadow: 0 10px 40px rgba(255, 107, 107, 0.3);
        }

        .results-normal {
            background: linear-gradient(135deg, #51cf66, #40c057);
            color: white;
            box-shadow: 0 10px 40px rgba(81, 207, 102, 0.3);
        }

        .result-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }

        .result-title {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .result-confidence {
            font-size: 3rem;
            font-weight: 900;
            margin: 20px 0;
            background: rgba(255,255,255,0.2);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }

        .result-recommendation {
            font-size: 1.1rem;
            line-height: 1.6;
            background: rgba(255,255,255,0.15);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }

        /* Modern Technical Analysis */
        .technical-analysis {
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(25px);
            padding: 32px;
            border-radius: 24px;
            margin: 32px 0;
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        }

        .analysis-title {
            text-align: center;
            font-size: 1.4rem;
            font-weight: 700;
            color: white;
            margin-bottom: 24px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 20px;
        }

        .analysis-item {
            text-align: center;
            padding: 24px 20px;
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(15px);
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.15);
            transition: all 0.3s ease;
        }

        .analysis-item:hover {
            background: rgba(255,255,255,0.12);
            transform: translateY(-2px);
        }

        .analysis-label {
            font-size: 0.85rem;
            color: rgba(255,255,255,0.7);
            font-weight: 600;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .analysis-value {
            font-size: 1.3rem;
            font-weight: 800;
            color: white;
            text-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }

        /* Modern Glassmorphic Footer */
        .footer {
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(25px);
            margin-top: 80px;
            padding: 48px;
            border-radius: 28px;
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        }

        .disclaimer {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 32px;
            border-bottom: 2px solid rgba(255,255,255,0.15);
        }

        .disclaimer h3 {
            color: white;
            margin-bottom: 16px;
            font-size: 1.4rem;
            font-weight: 700;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .disclaimer p {
            color: rgba(255,255,255,0.85);
            line-height: 1.7;
            font-size: 1rem;
            font-weight: 500;
        }

        .developer-info {
            text-align: center;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(15px);
            color: white;
            padding: 32px;
            border-radius: 20px;
            margin: 24px 0;
            border: 1px solid rgba(255,255,255,0.15);
        }

        .developer-name {
            font-size: 1.6rem;
            font-weight: 800;
            margin-bottom: 8px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .developer-title {
            font-size: 1.1rem;
            opacity: 0.9;
            margin-bottom: 6px;
            font-weight: 600;
        }

        .developer-subtitle {
            font-size: 1rem;
            opacity: 0.8;
            font-weight: 500;
            margin-bottom: 15px;
        }

        .social-links {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
        }

        .social-link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 50px;
            height: 50px;
            background: rgba(255,255,255,0.15);
            border-radius: 50%;
            transition: all 0.3s ease;
            font-size: 1.5rem;
            color: white;
            text-decoration: none;
        }

        .social-link:hover {
            transform: translateY(-5px);
            background: rgba(255,255,255,0.25);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .copyright {
            text-align: center;
            color: rgba(255,255,255,0.7);
            font-size: 0.95rem;
            margin-top: 24px;
            padding-top: 24px;
            border-top: 1px solid rgba(255,255,255,0.15);
            font-weight: 500;
        }

        /* Loading Spinner */
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
            margin-right: 10px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .container { padding: 0 15px; }
            .header { padding: 40px 0 30px; }
            .stats-grid { grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
            .upload-section { padding: 30px 20px; }
            .upload-area { padding: 30px 20px; }
            .footer { padding: 30px 20px; }
        }

        @media (max-width: 480px) {
            .stats-grid { grid-template-columns: 1fr; }
            .analysis-grid { grid-template-columns: 1fr; }
            .result-confidence { font-size: 2.5rem; }
            .social-links {
                gap: 15px;
            }
            .social-link {
                width: 45px;
                height: 45px;
                font-size: 1.3rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div class="logo">🫁</div>
            <h1 class="main-title">PneumoDetect AI</h1>
            <p class="subtitle">Clinical-Grade Artificial Intelligence</p>
            <div class="hero-description">
                Fast. Accurate. Reliable. AI-powered pneumonia detection - externally validated on 400+ scans
            </div>
        </header>

        <!-- Stats Grid -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">86%</div>
                <div class="stat-label">🎯 Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">96.4%</div>
                <div class="stat-label">🔍 Sensitivity</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">2.5s</div>
                <div class="stat-label">⏱ Avg. Prediction Time</div>
            </div>
        </div>

        <!-- Upload Section -->
        <div class="upload-section">
            <h2 class="upload-title">📤 Upload Chest X-Ray for Instant AI Analysis</h2>
            <div class="upload-area" id="uploadArea">
                <input type="file" class="file-input" id="fileInput" accept=".jpg,.jpeg,.png">
                <div class="upload-icon">🫁</div>
                <div class="upload-text">Drag & drop your file or click to browse</div>
                <div class="upload-subtext">Supported formats: JPG, PNG, JPEG | Max 200MB</div>
            </div>
            
            <div class="image-preview" id="imagePreview" style="display: none;">
                <img id="previewImg" class="preview-image" alt="Uploaded X-Ray">
                <p style="margin-top: 15px; color: #666; font-weight: 600;">📸 Uploaded Chest X-Ray - Ready for AI Analysis</p>
            </div>

            <button class="analyze-btn" id="analyzeBtn" disabled>🔬 Analyze X-ray with AI</button>
        </div>

        <!-- Results Section -->
        <div id="resultsSection" style="display: none;"></div>

        <!-- Footer -->
        <footer class="footer">
            <div class="disclaimer">
                <h3>⚠ Medical Disclaimer</h3>
                <p>
                    This AI tool is intended for <strong>preliminary screening purposes only</strong>.<br>
                    Always seek advice from qualified healthcare professionals before making medical decisions.
                </p>
            </div>

            <div class="developer-info">
                <div class="developer-name">👩‍💻 Ayushi Rathour</div>
                <div class="developer-title">Biotechnology Graduate | AI Healthcare Innovator</div>
                <div class="developer-subtitle">Trained, developed and deployed this model</div>
                
                <div class="social-links">
                    <a href="https://github.com/ayushirathour" target="_blank" class="social-link">
                        <i class="fab fa-github"></i>
                    </a>
                    <a href="https://huggingface.co/ayushirathour" target="_blank" class="social-link">
                        <i class="fas fa-robot"></i>
                    </a>
                </div>
            </div>

            <div class="copyright">
                <strong>PneumoDetect AI v2.0</strong> | © 2025 Ayushi Rathour<br>
                For model details and source code, visit GitHub
            </div>
        </footer>
    </div>

    <script>
        // H5 MODEL BACKEND API CONFIGURATION - YOUR EXISTING LOGIC
        const API_CONFIG = {
            endpoints: [
                '/api/predict',
                'http://localhost:5000/api/predict',
                'https://your-app.herokuapp.com/api/predict',
                'https://your-app.onrender.com/api/predict',
                'https://your-model-api.vercel.app/api/predict'
            ],
            timeout: 30000,
            maxFileSize: 200 * 1024 * 1024
        };

        // YOUR EXISTING H5 MODEL PREDICTION FUNCTION
        async function predictWithH5Model(imageFile) {
            const formData = new FormData();
            formData.append('image', imageFile);

            for (const endpoint of API_CONFIG.endpoints) {
                try {
                    console.log(`Trying API endpoint: ${endpoint}`);
                    
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        body: formData,
                        timeout: API_CONFIG.timeout
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }

                    const result = await response.json();
                    console.log('✅ Prediction successful from:', endpoint);
                    return result;

                } catch (error) {
                    console.log(`❌ Failed endpoint ${endpoint}:`, error.message);
                    continue;
                }
            }

            throw new Error('All API endpoints failed. Please check server connection.');
        }

        // YOUR EXISTING PREDICTION INTERPRETATION LOGIC
        function interpretPrediction(apiResponse) {
            let predictionScore;
            
            if (typeof apiResponse.prediction === 'number') {
                predictionScore = apiResponse.prediction;
            } else if (apiResponse.confidence) {
                predictionScore = apiResponse.confidence;
            } else if (apiResponse.probability) {
                predictionScore = apiResponse.probability;
            } else {
                predictionScore = 0.5;
            }

            let diagnosis, confidence, confidenceLevel, recommendation;
            
            if (predictionScore > 0.5) {
                diagnosis = "PNEUMONIA";
                confidence = predictionScore * 100;
                
                if (confidence >= 80) {
                    confidenceLevel = "High";
                    recommendation = "🚨 Strong indication — seek immediate medical attention.";
                } else if (confidence >= 60) {
                    confidenceLevel = "Moderate";
                    recommendation = "⚠ Moderate indication — medical review advised.";
                } else {
                    confidenceLevel = "Low";
                    recommendation = "💡 Possible pneumonia — further examination advised.";
                }
            } else {
                diagnosis = "NORMAL";
                confidence = (1 - predictionScore) * 100;
                
                if (confidence >= 80) {
                    confidenceLevel = "High";
                    recommendation = "✅ No signs of pneumonia detected — chest appears normal.";
                } else if (confidence >= 60) {
                    confidenceLevel = "Moderate";
                    recommendation = "👍 Likely normal — routine follow-up if symptoms persist.";
                } else {
                    confidenceLevel = "Low";
                    recommendation = "🤔 Unclear result — manual review recommended.";
                }
            }

            return {
                diagnosis,
                confidence: Math.round(confidence * 100) / 100,
                confidenceLevel,
                recommendation,
                rawScore: predictionScore
            };
        }

        // File upload functionality
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const imagePreview = document.getElementById('imagePreview');
        const previewImg = document.getElementById('previewImg');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const resultsSection = document.getElementById('resultsSection');

        let uploadedFile = null;

        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please upload an image file (JPG, PNG, JPEG)');
                return;
            }

            if (file.size > API_CONFIG.maxFileSize) {
                alert('File size exceeds 200MB limit. Please upload a smaller image.');
                return;
            }

            uploadedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                imagePreview.style.display = 'block';
                analyzeBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }

        // YOUR EXISTING H5 MODEL ANALYSIS LOGIC
        analyzeBtn.addEventListener('click', async () => {
            if (!uploadedFile) {
                alert('Please upload an image first');
                return;
            }

            analyzeBtn.innerHTML = '<div class="loading"></div>🧠 Advanced AI analysis in progress...';
            analyzeBtn.disabled = true;

            try {
                const apiResponse = await predictWithH5Model(uploadedFile);
                const result = interpretPrediction(apiResponse);
                displayResults(result, result.rawScore);
                
            } catch (error) {
                console.error('Error during H5 model prediction:', error);
                alert(`Analysis failed: ${error.message}\n\nPlease ensure the backend server is running.`);
            }
            
            analyzeBtn.innerHTML = '🔬 Analyze X-ray with AI';
            analyzeBtn.disabled = false;
        });

        // COMPLETED displayResults FUNCTION
        function displayResults(result, prediction) {
            const isPneumonia = result.diagnosis === 'PNEUMONIA';
            const resultClass = isPneumonia ? 'results-pneumonia' : 'results-normal';
            const resultIcon = isPneumonia ? '🩺' : '✅';
            
            resultsSection.innerHTML = `
                <div class="results-section ${resultClass}">
                    <div class="result-icon">${resultIcon}</div>
                    <div class="result-title">Diagnosis Result: ${result.diagnosis === 'PNEUMONIA' ? 'Pneumonia Detected' : 'Normal Chest X-Ray'}</div>
                    <div class="result-confidence">${result.confidence}% confidence</div>
                    <div style="font-size: 1.2rem; margin: 15px 0; font-weight: 700;">
                        Confidence Level: ${result.confidenceLevel}
                    </div>
                    <div class="result-recommendation">${result.recommendation}</div>
                </div>

                <div class="technical-analysis">
                    <div class="analysis-title">🔬 Technical Analysis Dashboard</div>
                    <div class="analysis-grid">
                        <div class="analysis-item">
                            <div class="analysis-label">Model Architecture</div>
                            <div class="analysis-value">MobileNetV2</div>
                        </div>
                        <div class="analysis-item">
                            <div class="analysis-label">Threshold</div>
                            <div class="analysis-value">0.5</div>
                        </div>
                        <div class="analysis-item">
                            <div class="analysis-label">Raw Score</div>
                            <div class="analysis-value">${prediction.toFixed(4)}</div>
                        </div>
                    </div>
                </div>
            `;
            
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        // Add click event to upload area
        uploadArea.addEventListener('click', (e) => {
            if (e.target !== fileInput) {
                fileInput.click();
            }
        });

        // Check API availability on page load
        window.addEventListener('load', async () => {
            console.log('🔍 Checking API availability...');
            try {
                const response = await fetch('/api/health', { method: 'GET' });
                if (response.ok) {
                    console.log('✅ Backend API is available');
                } else {
                    console.log('⚠️ Backend API health check failed');
                }
            } catch (error) {
                console.log('⚠️ Backend API not reachable:', error.message);
            }
        });
    </script>
</body>
</html>
