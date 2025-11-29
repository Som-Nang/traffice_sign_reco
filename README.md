# Traffic Sign Recognition AI - Web Interface

A professional web-based GUI for traffic sign recognition using deep learning. This application allows users to upload traffic sign images and get instant AI-powered predictions with confidence scores.

## Features

- 🎨 **Modern, Professional UI** - Beautiful gradient design with smooth animations
- 📤 **Drag & Drop Upload** - Easy image upload with drag-and-drop support
- 🔍 **Real-time Analysis** - Instant AI predictions with confidence scores
- 📊 **Top 5 Predictions** - See the top 5 most likely traffic signs
- 📱 **Responsive Design** - Works perfectly on desktop, tablet, and mobile
- 🌐 **Multi-language Support** - Shows traffic sign names in English and Khmer

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd traffic_sign_ai
   ```

```bash
source /venv/bin/activate
```

`````bash
pip install -r requirements.txt
````bash
1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

1. **Start the Flask Server**

   ```bash
   python app.py
   ```

2. **Open Your Browser**
   Navigate to:

   ```
   http://localhost:5000
   ```

3. **Upload and Analyze**
   - Drag and drop a traffic sign image, or click to browse
   - Click "Analyze Traffic Sign" button
   - View the AI prediction results with confidence scores

## Run with Docker-Compose

Install Docker and Docker-Compose, then run:

1. **Build and Start the Docker Containers**

   ```bash
   docker-compose up -d
   ```

## Application Structure

```
traffic_sign_ai/
├── app.py                  # Flask backend server
├── templates/
│   └── index.html          # Web interface (HTML + CSS + JavaScript)
├── uploads/                # Uploaded images (created automatically)
├── model/
│   └── traffic_sign_model.h5  # Trained model
├── predict.py              # Command-line prediction script
├── model.py                # Model architecture
├── preprocess.py           # Image preprocessing
└── requirements.txt        # Python dependencies
```

## API Endpoints

### POST /predict

Upload an image and get prediction results.

**Request:**

- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**

```json
{
  "success": true,
  "class_id": 14,
  "sign_name": "Stop : ឈប់",
  "confidence": 99.87,
  "confidence_level": "Very High",
  "confidence_message": "Reliable prediction",
  "confidence_status": "excellent",
  "top_5_predictions": [
    {
      "class_id": 14,
      "sign_name": "Stop : ឈប់",
      "confidence": 99.87
    },
    ...
  ]
}
```

## Confidence Levels

- **Very High (≥90%)**: ✅ Reliable prediction
- **Good (70-89%)**: ⚠️ Fairly reliable
- **Moderate (50-69%)**: ⚠️ Use with caution
- **Low (<50%)**: ❌ Unreliable prediction

## Supported Traffic Signs

The model recognizes 44 different types of traffic signs from the GTSRB dataset, including:

- Speed limits (20-120 km/h)
- Warning signs (curves, pedestrians, children, etc.)
- Mandatory signs (turn directions, roundabouts, etc.)
- Prohibitory signs (no entry, no passing, stop, yield, etc.)

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **AI/ML**: TensorFlow/Keras
- **Image Processing**: OpenCV, NumPy

## Command-Line Usage (Alternative)

You can still use the command-line interface:

```bash
python predict.py
```

This will predict the sign in `test_sign.png` and display results in the terminal.

## Troubleshooting

- **Port 5000 already in use**: Change the port in `app.py` (last line)
- **Model not found**: Ensure `model/traffic_sign_model.h5` exists
- **Image upload fails**: Check that the image is a valid JPG/PNG file
- **Slow predictions**: First prediction may be slow due to model loading

## Browser Compatibility

- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Opera

## License

This project uses the GTSRB (German Traffic Sign Recognition Benchmark) dataset.
`````
