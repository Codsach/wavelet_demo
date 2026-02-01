import os
import cv2
import numpy as np
import pywt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# Wavelet Packet Feature Extraction (BULLETPROOF)
# -------------------------------------------------
def extract_wavelet_packet_features(image):
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    wp = pywt.WaveletPacket2D(
        data=gray,
        wavelet='db1',
        mode='symmetric',
        maxlevel=2
    )

    features = []

    # Explicitly define level-2 paths (NO get_level used)
    level2_paths = [
        'aa', 'ad', 'da', 'dd',
        'aa', 'ad', 'da', 'dd'
    ]

    # Actually valid level-2 paths
    level2_paths = ['aa', 'ad', 'da', 'dd']

    for path in level2_paths:
        coeffs = wp[path].data
        features.append(np.mean(coeffs))
        features.append(np.std(coeffs))
        features.append(np.var(coeffs))

    return features


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
X = []
y = []

dataset_path = "dataset"

for label, folder in enumerate(["class0", "class1"]):
    folder_path = os.path.join(dataset_path, folder)

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        image = cv2.imread(img_path)

        if image is None:
            continue

        features = extract_wavelet_packet_features(image)
        X.append(features)
        y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# -------------------------------------------------
# Train-Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------------------------
# Train SVM
# -------------------------------------------------
model = SVC(kernel='rbf', gamma='scale')
model.fit(X_train, y_train)

# -------------------------------------------------
# Predict & Accuracy
# -------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Classification Accuracy:", round(accuracy * 100, 2), "%")