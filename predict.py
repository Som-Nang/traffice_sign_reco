import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess

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
    13: 'Yield : បញ្ឈប់ឲ្យផ្លូវស្របទិស',
    14: 'Stop : ឈប់',
    15: 'No vehicles : មិនអនុញ្ញាតយានយន្ត',
    16: 'Vehicles over 3.5 metric tons prohibited : មិនអនុញ្ញាតយានយន្តលើស 3.5 តោន',
    17: 'No entry : មិនអាចចូល',
    18: 'General caution : ប្រយ័ត្នទូទៅ',
    19: 'Dangerous curve to the left : ជ្រុងឈ្វេងខាងឆ្វេងគួរប្រុងប្រយ័ត្ន',
    20: 'Dangerous curve to the right : ជ្រុងឈ្វេងខាងស្ដាំគួរប្រុងប្រយ័ត្ន',
    21: 'Double curve : ជ្រុងឈ្វេងទ្វេ',
    22: 'Bumpy road : ផ្លូវខូច',
    23: 'Slippery road : ផ្លូវរល់',
    24: 'Road narrows on the right : ផ្លូវស្ដាំតូច',
    25: 'Road work : ការងារផ្លូវ',
    26: 'Traffic signals : អំពូលសញ្ញាចរាចរ',
    27: 'Pedestrians : ជើងដើរ',
    28: 'Children crossing : កុមារឆ្លងផ្លូវ',
    29: 'Bicycles crossing : កង់ឆ្លងផ្លូវ',
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
    43: 'No left u-turn : មិនអនុញ្ញាតបត់ស្ដាំ',
}

model = load_model("model/traffic_sign_model.keras")

def predict_sign(path):
    img = cv2.imread(path)

    if img is None:
        print("❌ Error: Image not found!")
        return

    img = cv2.resize(img, (32, 32))
    img = preprocess(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)
    sign_name = class_names.get(class_id, "Unknown sign")

    # Display results
    print("\n" + "="*50)
    print("🚦 TRAFFIC SIGN PREDICTION RESULTS")
    print("="*50)
    print(f"📁 Image: {path}")
    print(f"🛑 Predicted Class ID: {class_id}")
    print(f"🏷️  Sign Name: {sign_name}")
    print(f"📊 Confidence: {round(confidence * 100, 2)}%")
    
    # Show confidence level interpretation
    if confidence >= 0.9:
        print("✅ Confidence Level: Very High - Reliable prediction")
    elif confidence >= 0.7:
        print("⚠️  Confidence Level: Good - Fairly reliable")
    elif confidence >= 0.5:
        print("⚠️  Confidence Level: Moderate - Use with caution")
    else:
        print("❌ Confidence Level: Low - Unreliable prediction")
    
    print("="*50 + "\n")

# Example
predict_sign("test_sign.png")
