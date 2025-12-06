
# Emotion Detection AI

- [Emotion Detection AI](#emotion-detection-ai)
  - [Project Title](#project-title)
  - [Introduction](#introduction)
  - [Objectives](#objectives)
  - [Tools and Technologies Used](#tools-and-technologies-used)
    - [Python](#python)
    - [NumPy](#numpy)
    - [Pillow](#pillow)
    - [scikit-learn](#scikit-learn)
    - [joblib](#joblib)
    - [scikit-image](#scikit-image)
  - [Project Structure](#project-structure)
  - [Modules \& Functionalities](#modules--functionalities)
  - [Feature Extraction (HOG)](#feature-extraction-hog)
  - [Training \& Loading Model](#training--loading-model)
  - [Main Program](#main-program)
  - [Usage](#usage)
  - [Requirements](#requirements)
  - [Sample Output](#sample-output)
  - [Conclusion](#conclusion)

## Project Title

**Emotion Detection AI**  

A Python-based system that detects human facial emotions from images. It extracts features from images using **HOG (Histogram of Oriented Gradients)** and classifies them into emotions like angry, disgust, fear, happy, neutral, sad, and surprise using **SGDClassifier**.

## Introduction

Human emotions can be detected from facial expressions. This project uses machine learning to:

- Analyze grayscale face images.
- Extract features using **HOG**.
- Train a model to classify facial emotions.
- Predict emotions for single images or entire folders.

It provides an easy-to-use terminal interface for predictions.

## Objectives

- Detect emotions in face images accurately.
- Allow predictions on a single image or folder of images.
- Save and load trained models to avoid retraining every time.
- Keep the project lightweight and fast by limiting images per class.

## Tools and Technologies Used

### Python

The core programming language for this project.

### NumPy

Used for numerical computations and handling image arrays.

### Pillow

Loads and processes images (resizing and converting to grayscale).

### scikit-learn

Provides **SGDClassifier** for training the emotion detection model and **LabelEncoder** for encoding class labels.

### joblib

Saves and loads trained models efficiently.

### scikit-image

Provides **HOG (Histogram of Oriented Gradients)** feature extraction for images.

## Project Structure

emotion-detection/
│
├─ data_set/
│  ├─ train/
│  │  ├─ angry/
│  │  ├─ disgust/
│  │  ├─ fear/
│  │  ├─ happy/
│  │  ├─ neutral/
│  │  ├─ sad/
│  │  └─ surprise/
│
├─ models/
│  └─ emotion_model.pkl
│
├─ main.py
└─ requirements.txt

## Modules & Functionalities

- **main.py** – Main script to train, load, and predict emotions.
- **models/** – Stores the trained model.
- **data_set/** – Folder containing images organized by emotion classes.

## Feature Extraction (HOG)

- Converts grayscale images into **feature vectors**.
- Captures **edges and gradients** to detect facial structures.
- HOG features are used as input for the classifier.

## Training & Loading Model

- Loads dataset from `data_set/train`.
- Trains **SGDClassifier** if no model exists.
- Saves trained model to `models/emotion_model.pkl`.
- Loads model automatically on subsequent runs.

## Main Program

- Terminal-based interface:
  1. Enter path of a single image or folder.
  2. Predictions are shown for each image.
  3. Type `exit` to quit.

- Handles both **single image** and **folder** predictions.
- Automatically converts Windows paths to a universal format.

## Usage

Activate virtual environment:

```bash
& C:/Users/user1/Desktop/emotion-detection/venv/Scripts/Activate.ps1
````

Install requirements

```bash
pip install -r requirements.txt
```

1. Run the program:

```bash
python main.py
```

Enter image path or folder path when prompted:

```text
Enter image path or folder (or 'exit'): C:/Users/user1/Desktop/emotion-detection/data_set/train/test/disgust
```

1. Type `exit` to quit.

## Requirements

Include in `requirements.txt`:

numpy==1.24.3

Pillow==12.0.0

scikit-learn==1.7.2

joblib==1.5.2

scikit-image==0.21.0

**numpy** – Numerical operations
**Pillow** – Image processing
**scikit-learn** – Machine learning (SGDClassifier, LabelEncoder)
**joblib** – Save/load model
**scikit-image** – HOG feature extraction

Install all requirements using:

```bash
pip install -r requirements.txt
```

## Sample Output

```text
Enter image path or folder (or 'exit'): C:/.../disgust
image1.jpg → disgust
image2.jpg → neutral
image3.jpg → happy
...
```

## Conclusion

This project demonstrates a **simple, lightweight AI system** for detecting facial emotions.
It is easy to set up, train, and run on any system using Python and terminal commands.
