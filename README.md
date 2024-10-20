
# Smart Quality Test System for E-Commerce Shipment using Camera Vision Technology (Smart Vision Technology Quality Control)

### Project Overview
This project, **Smart Quality Test System for E-Commerce Shipment using Camera Vision Technology**, focuses on leveraging AI and computer vision techniques to detect product details, quality, and freshness in an automated manner, particularly for e-commerce shipments. The system processes various product types like groceries, cosmetics, and perishables (fruits and vegetables) to ensure that the shipped items meet quality standards.

---


## Dataset

The trained dataset for product recognition and freshness detection is uploaded to Google Drive. Due to its size (greater than 25 MB), you can access it through the following 
- [Product Recognition Dataset](https://drive.google.com/drive/folders/1cQCXSOCLoZMnYMYAwtAHbcSCiV49DM4w?usp=sharing)
- [Fruits and Vegetables Freshness Dataset](https://drive.google.com/drive/folders/1XCyvOUaNa6Q4PBwzwREW3tfDYsb0IcG7?usp=sharing).

---

### How to Run the Project:

1. **Install Dependencies**:
   Make sure to install all the required dependencies using the following command:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask Application**:
   Start the Flask server by running:

   ```bash
   python app.py
   ```

3. **Access the Application**:
   Once the server is running, open your browser and go to:

   ```
   http://127.0.0.1:5000
   ```

4. **Upload Image**:
   Upload the image of the product you want to analyze, and the system will output the product details, quality, and freshness level.

---

## Tech Stack

- **Easy OCR**: Used for extracting text-based information like brand name, product description, and expiry date from product images.
- **TensorFlow**: Deep learning framework used for training the product recognition models.
- **OpenCV**: Used for image processing tasks like product detection and assessing the freshness of fruits and vegetables.
- **Flask Framework**: Used to build the backend and serve the application.
- **Matplotlib**: Used for data visualization and model performance analysis.
  
---

## Process Flow

### 1. **Data Collection & Preparation**
   - We collected a diverse set of product data, including **cosmetics, groceries, and other household items**, for the training process. 
   - The sample images were annotated using **Roboflow** to extract critical information like:
     - **Brand name**
     - **Product description**
     - **Expiry date**

### 2. **Model Training**
   - The collected data was processed and trained using **TensorFlow** for product recognition and classification.
   - **EasyOCR** was integrated into the system to extract textual information such as the brand and expiry date from product packaging.
   - **OpenCV** was used for image processing tasks, including feature extraction from product images.

### 3. **Freshness Detection**
   - We used **OpenCV** to implement an algorithm for detecting the **freshness** of fruits and vegetables. By using a specific dataset for fresh and stale produce, the system can classify items as **fresh** or **stale**.
   
### 4. **IR-Based Product Counting**
   - For counting the number of products, we utilized an **Arduino Nano microcontroller**. The **IR sensor** counts the items as they pass through the camera.
   - This feature allows efficient stock tracking and is a key component of our conveyor belt extension plans.

### 5. **Frontend Design**
   - The user interface for this system was developed using **HTML** and **CSS**, providing a simple and intuitive user experience.
   - The **Flask framework** serves as the backend, connecting the user interface to the trained models and processing the incoming data from the camera.

---
