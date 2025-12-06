import os
import numpy as np
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import random
from skimage.feature import hog

# =========================
# SETTINGS
# =========================
TRAIN_DIR = "data_set/train"
MODEL_PATH = "models/emotion_model.pkl"
IMG_SIZE = (48, 48)
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
MAX_IMAGES_PER_CLASS = 50  # small for fast training

# =========================
# FEATURE EXTRACTION
# =========================
def extract_hog_features(img_path):
    img = Image.open(img_path).convert("L").resize(IMG_SIZE)
    img_np = np.array(img)
    features = hog(img_np,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   orientations=9,
                   block_norm='L2-Hys')
    return features

# =========================
# DATA LOADING
# =========================
def load_dataset(base_path):
    X, y = [], []
    for cls in CLASSES:
        folder = os.path.join(base_path, cls)
        if os.path.isdir(folder):
            imgs = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            if len(imgs) > MAX_IMAGES_PER_CLASS:
                imgs = random.sample(imgs, MAX_IMAGES_PER_CLASS)
            for img_name in imgs:
                try:
                    features = extract_hog_features(os.path.join(folder, img_name))
                    X.append(features)
                    y.append(cls)
                except:
                    continue
    X = np.array(X)
    y = np.array(y)
    return X, y

# =========================
# TRAIN OR LOAD MODEL
# =========================
def get_model():
    if os.path.exists(MODEL_PATH):
        print("Loading model...")
        return joblib.load(MODEL_PATH)
    print("Training new model...")
    X_train, y_train = load_dataset(TRAIN_DIR)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    clf = SGDClassifier(max_iter=5000, tol=1e-3, loss='log_loss', random_state=42)
    clf.fit(X_train, y_encoded)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump((clf, le), MODEL_PATH)
    print("Model saved.")
    return clf, le

# =========================
# PREDICT SINGLE IMAGE
# =========================
def predict_image(clf, le, img_path):
    try:
        features = extract_hog_features(img_path).reshape(1, -1)
        pred = clf.predict(features)[0]
        emotion = le.inverse_transform([pred])[0]
        print(f"{img_path} → {emotion}")
    except Exception as e:
        print(f"Error: {img_path} | {e}")

# =========================
# MAIN LOOP (single image or folder)
# =========================
if __name__ == "__main__":
    clf, le = get_model()
    while True:
        user_input = input("\nEnter image path or folder (or 'exit'): ").strip()
        if user_input.lower() == "exit":
            break

        # Normalize Windows paths
        path = os.path.expanduser(user_input)
        path = os.path.abspath(path)
        path = path.replace("\\", "/")  # forward slashes work on Windows

        if os.path.isfile(path):
            predict_image(clf, le, path)
        elif os.path.isdir(path):
            # predict all images in the folder
            files = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            if not files:
                print("No images found in folder:", path)
            for img_name in files:
                predict_image(clf, le, os.path.join(path, img_name))
        else:
            print("Invalid path. Make sure you use correct slashes or check the path:", path)
