from flask import Flask, request, render_template, jsonify
import os
import json
import re
import numpy as np
import easyocr
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import cv2
import threading
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Load the saved models and class indices for product recognition
model = load_model('my_model.keras')
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
label_map = {v: k for k, v in class_indices.items()}

# Load the dataset for freshness detection
dataset_dir = r'C:\Users\uvara\OneDrive\Desktop\lets begin\Brandrecognition2\env\dataset'

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Global variable to control the live detection loop
is_live_detection_active = False

# Preprocess images for brand prediction
def preprocess_image(image_path):
    img_array = cv2.imread(image_path)
    # If the input is a grayscale image, convert it to 3 channels (RGB)
    if len(img_array.shape) == 2:  # Grayscale image (no channels dimension)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 1:  # Single channel image with explicit channels dimension
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    img_array = cv2.resize(img_array, (224, 224))  # Resize the image to 224x224
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values
    return img_array

# Function to extract product description using EasyOCR (updated, without expiry date)
def extract_text_details(image_path):
    result = reader.readtext(image_path, detail=1, paragraph=False)

    description = ""
    # Loop through OCR results and gather descriptions
    for detection in result:
        text = detection[1]
        confidence = detection[2]

        # Only consider text with confidence > 0.5
        if confidence > 0.5:
            description += text + " "

    return description.strip()

def extract_color_histogram(image):
    """Extract color histogram features from the image."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Load dataset for freshness detection
def load_dataset(dataset_path):
    features = []
    labels = []

    for fruit_type in os.listdir(dataset_path):
        for freshness in ['fresh', 'stale']:
            fruit_folder = os.path.join(dataset_path, fruit_type, freshness)
            for image_name in os.listdir(fruit_folder):
                image_path = os.path.join(fruit_folder, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Image not found or unable to read: {image_path}")
                hist = extract_color_histogram(image)
                features.append(hist)
                labels.append(freshness)

    return np.array(features), np.array(labels)

# Load the dataset for freshness detection
features, labels = load_dataset(dataset_dir)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardize features for better SVM performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM classifier
model_freshness = SVC(kernel='linear', probability=True)
model_freshness.fit(X_train, y_train)

# Route for main HTML page
@app.route('/')
def product_recognition():
    return render_template('product_recognition.html')

@app.route('/fruits')
def fruits_freshness():
    return render_template('fruits_freshness.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'front_image' not in request.files or 'back_image' not in request.files:
        return jsonify({'error': 'Both front and back images are required'})

    front_image = request.files['front_image']
    back_image = request.files['back_image']

    if front_image.filename == '' or back_image.filename == '':
        return jsonify({'error': 'Both images must be selected'})

    front_image_path = os.path.join('Branddataset', secure_filename(front_image.filename))
    back_image_path = os.path.join('Branddataset', secure_filename(back_image.filename))

    try:
        # Save front image
        front_image.save(front_image_path)

        # Save back image
        back_image.save(back_image_path)

        # Preprocess and make brand prediction (from front image)
        img_array = preprocess_image(front_image_path)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_brand = label_map.get(predicted_class, "Unknown")

        # Extract product description (from back image)
        description = extract_text_details(back_image_path)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed due to an error'})

    finally:
        try:
            if os.path.exists(front_image_path):
                os.remove(front_image_path)
            if os.path.exists(back_image_path):
                os.remove(back_image_path)
        except PermissionError as e:
            print(f"Permission error while deleting file: {e}")

    return jsonify({
        'predicted_brand': predicted_brand,
        'description': description
    })

# Route for starting live detection of fruit freshness
@app.route('/live_detection', methods=['GET'])
def live_detection():
    global is_live_detection_active
    is_live_detection_active = True
    camera_index = 2
    threading.Thread(target=live_detection_with_bounding_box, args=(camera_index,), daemon=True).start()
    return jsonify({'status': 'Live detection started'})

# Route for stopping live detection
@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global is_live_detection_active
    is_live_detection_active = False
    return jsonify({'status': 'Live detection stopped'})

def live_detection_with_bounding_box(camera_index=2):
    global is_live_detection_active
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Cannot access the camera stream")
        return

    while is_live_detection_active:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, 100, 100])  # HSV range for red color
        upper_bound = np.array([10, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area to be considered a valid fruit region
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract the fruit region for classification
                fruit_region = frame[y:y + h, x:x + w]

                # Preprocess the fruit region for freshness detection
                hist = extract_color_histogram(fruit_region)
                hist = scaler.transform([hist])  # Scale the histogram features

                # Predict freshness using the SVM model
                freshness_prediction = model_freshness.predict(hist)
                freshness_label = label_encoder.inverse_transform(freshness_prediction)[0]

                # Display the freshness label on the video feed
                cv2.putText(frame, f"Freshness: {freshness_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Show the live detection window
        cv2.imshow("Live Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True)
