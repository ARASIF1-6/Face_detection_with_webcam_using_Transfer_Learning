# 🎥 Face Detection with Webcam using Transfer Learning

A modern and efficient **real-time face detection** system built using **Transfer Learning** and **Deep Learning**.
This project fine-tunes a pre-trained model to detect human faces from a live webcam feed with high accuracy.

---

## 🚀 Project Overview

This project demonstrates how **transfer learning** can be used to create a powerful face detection model with minimal data and training time.
It uses a **pre-trained CNN (like MobileNetV2 or ResNet50)** as the base model, and then fine-tunes it for face vs non-face classification.

Once trained, the model can perform **real-time detection** through a webcam, drawing bounding boxes around detected faces.

---

## 🧠 Key Features

✅ Real-time face detection using webcam
✅ Transfer learning-based training for faster and more accurate results
✅ Lightweight and easy to deploy
✅ Jupyter Notebook included for custom model training
✅ Highly extensible (face recognition, attendance, etc.)

---

## 📂 Repository Structure

```
Face_detection_with_webcam_using_Transfer_Learning/
│
├── train_transfer_learning.ipynb   # Train / Fine-tune the model using transfer learning
├── detect_webcam.py                # Run real-time face detection with webcam
├── models/                         # Store pre-trained and trained model weights
├── assets/                         # Optional: demo images, screenshots, or gifs
└── README.md                       # Project documentation
```

---

## 🛠 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ARASIF1-6/Face_detection_with_webcam_using_Transfer_Learning.git
cd Face_detection_with_webcam_using_Transfer_Learning
```

### 2️⃣ (Optional) Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # For macOS/Linux
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install Dependencies

```bash
pip install opencv-python tensorflow numpy matplotlib
```

---

## 🧩 How to Use

### 🔹 Step 1: Train / Fine-tune the Model

Open the Jupyter Notebook:

```bash
jupyter notebook train_transfer_learning.ipynb
```

Inside the notebook:

* Load a pre-trained model (e.g., MobileNetV2, ResNet50)
* Add custom layers for binary classification (face / non-face)
* Train on your dataset
* Save the trained model (e.g., `model_face.h5`)

---

### 🔹 Step 2: Run the Real-time Detection

Once training is complete, or if you have a pre-trained model, simply run:

```bash
python detect_webcam.py
```

This will open your webcam and start detecting faces in real-time.

---

## ⚙️ Sample Code – detect_webcam.py

```python
import cv2
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model('models/model_face.h5')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for model input
    resized = cv2.resize(frame, (128, 128))
    img = np.expand_dims(resized / 255.0, axis=0)

    # Predict
    prediction = model.predict(img)
    if prediction[0][0] > 0.5:  # Threshold for 'face detected'
        cv2.putText(frame, 'Face Detected', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.rectangle(frame, (20, 20), (300, 300), (0,255,0), 2)

    # Display the frame
    cv2.imshow('Face Detection (Transfer Learning)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📊 Example Output

When you run the project, it will open your webcam and draw a green box around any detected faces.

| Input                                                           | Output                                                            |
| --------------------------------------------------------------- | ----------------------------------------------------------------- |
| ![Input](https://via.placeholder.com/300x200?text=Webcam+Input) | ![Output](https://via.placeholder.com/300x200?text=Detected+Face) |


---

