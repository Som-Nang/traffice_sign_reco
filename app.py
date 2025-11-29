from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess
import os
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Traffic sign class names (GTSRB dataset)
class_names = {
    0: 'Speed limit (20km/h) : គំរិតល្បឿន (20គ.ម/ម៉ោង)',
    1: 'Speed limit (30km/h) : គំរិតល្បឿន (30គ.ម/ម៉ោង)',
    2: 'Speed limit (50km/h) : គំរិតល្បឿន (50គ.ម/ម៉ោង)',
    3: 'Speed limit (60km/h) : គំរិតល្បឿន (60គ.ម/ម៉ោង)',
    4: 'Speed limit (70km/h) : គំរិតល្បឿន (70គ.ម/ម៉ោង)',
    5: 'Speed limit (80km/h) : គំរិតល្បឿន (80គ.ម/ម៉ោង)',
    6: 'End of speed limit (80km/h) : បញ្ចប់គំរិតល្បឿន (80គ.ម/ម៉ោង)',
    7: 'Speed limit (100km/h) : គំរិតល្បឿន (100គ.ម/ម៉ោង)',
    8: 'Speed limit (120km/h) : គំរិតល្បឿន (120គ.ម/ម៉ោង)',
    9: 'No passing : មិនអនុញ្ញាតឆ្លងកាត់',
    10: 'No passing for vehicles over 3.5 metric tons : មិនអនុញ្ញាតឆ្លងកាត់សម្រាប់យានយន្តលើស 3.5 តោន',
    11: 'Right-of-way at the next intersection : អាទិភាពនៅចំណុចឆ្លងបន្ទាប់',
    12: 'Priority road : ផ្លូវអាទិភាព',
    13: 'Yield : ផ្តល់អទិភាពអោយផ្លូវនៅត្រង់ចំនុចប្រសព្វ',
    14: 'Stop : ឈប់',
    15: 'No vehicles : មិនអនុញ្ញាតយានយន្ត',
    16: 'Vehicles over 3.5 metric tons prohibited : មិនអនុញ្ញាតយានយន្តលើស 3.5 តោន',
    17: 'No entry : មិនអាចចូល',
    18: 'General caution : ប្រយ័ត្នទូទៅ',
    19: 'Dangerous curve to the left : ផ្លូវកោងខ្លាំងទៅឆ្វេង',
    20: 'Dangerous curve to the right : ផ្លូវកោងខ្លាំងទៅស្ដាំ',
    21: 'Double curve : ជ្រុងឈ្វេងទ្វេ',
    22: 'Bumpy road : ផ្លូវខូច',
    23: 'Slippery road : ផ្លូវរអិល',
    24: 'Road narrows on the right : ផ្លូវស្ដាំតូច',
    25: 'Road work : ផ្លូវមានការដ្ធាន',
    26: 'Traffic signals : អំពូលសញ្ញាចរាចរ',
    27: 'Pedestrians : អ្នក​ថ្មើរជើង',
    28: 'Children crossing : ផ្លូវកុមារ',
    29: 'Bicycles crossing : ផ្លូវកង់',
    30: 'Beware of ice/snow : ប្រយ័ត្នទឹកកក/ព Schnee',
    31: 'Wild animals crossing : សត្វព្រៃឆ្លងផ្លូវ',
    32: 'End of all speed and passing limits : បញ្ចប់គំរិតល្បឿន និងឆ្លងកាត់ទាំងអស់',
    33: 'Turn right ahead : បត់ស្ដាំខាងមុខ',
    34: 'Turn left ahead : បត់ឆ្វេងខាងមុខ',
    35: 'Ahead only : ទៅមុខតែប៉ុណ្ណោះ',
    36: 'Go straight or right : ទៅត្រង់ឬស្ដាំ',
    37: 'Go straight or left : ទៅត្រង់ឬឆ្វេង',
    38: 'Keep right : រក្សាស្ដាំ',
    39: 'Keep left : រក្សាឆ្វេង',
    40: 'Roundabout mandatory : គួរតែចូលរង្វង់',
    41: 'End of no passing : បញ្ចប់ការមិនអនុញ្ញាតឆ្លងកាត់',
    42: 'End of no passing by vehicles over 3.5 metric tons : បញ្ចប់ការមិនអនុញ្ញាតឆ្លងកាត់សម្រាប់យានយន្តលើស 3.5 តោន',
    43: 'No left u-turn : មិនអនុញ្ញាតបត់ក្រោយ',
}

# Load model once at startup
print("Loading model...")
try:
    model = load_model("model/traffic_sign_model.keras")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please train the model first by running: docker-compose up model_trainer")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 503
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image file
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Preprocess image
        img_resized = cv2.resize(img, (32, 32))
        img_processed = preprocess(img_resized)
        img_input = np.expand_dims(img_processed, axis=0)
        
        # Make prediction
        prediction = model.predict(img_input, verbose=0)
        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        sign_name = class_names.get(class_id, "Unknown sign")
        
        # Determine confidence level
        if confidence >= 0.9:
            confidence_level = "Very High"
            confidence_message = "Reliable prediction"
            confidence_status = "excellent"
        elif confidence >= 0.7:
            confidence_level = "Good"
            confidence_message = "Fairly reliable"
            confidence_status = "good"
        elif confidence >= 0.5:
            confidence_level = "Moderate"
            confidence_message = "Use with caution"
            confidence_status = "moderate"
        else:
            confidence_level = "Low"
            confidence_message = "Unreliable prediction"
            confidence_status = "low"
        
        # Get top 5 predictions
        top_5_indices = np.argsort(prediction[0])[-5:][::-1]
        top_5_predictions = []
        for idx in top_5_indices:
            top_5_predictions.append({
                'class_id': int(idx),
                'sign_name': class_names.get(int(idx), "Unknown"),
                'confidence': float(prediction[0][idx]) * 100
            })
        
        response = {
            'success': True,
            'class_id': class_id,
            'sign_name': sign_name,
            'confidence': round(confidence * 100, 2),
            'confidence_level': confidence_level,
            'confidence_message': confidence_message,
            'confidence_status': confidence_status,
            'top_5_predictions': top_5_predictions
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
