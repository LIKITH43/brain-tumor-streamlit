"""
train.py  –  Train a brain-tumor classifier from a local dataset and save model.pkl.

Expected dataset layout (Kaggle "Brain Tumor MRI Dataset" works directly):
    data/
        tumor/      ← MRI images WITH tumors  (jpg/png)
        no_tumor/   ← MRI images WITHOUT tumors

Usage:
    python train.py --data_dir data/ --output model.pkl

Install deps first:
    pip install -r requirements.txt
"""

import argparse
import os
import sys
import numpy as np
import cv2
import joblib
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 128


def extract_features(img: Image.Image) -> np.ndarray:
    gray = np.array(img.convert("L").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    norm = gray / 255.0

    feats = [
        norm.mean(), norm.std(),
        np.percentile(norm, 25), np.percentile(norm, 75),
        norm.max() - norm.min(),
    ]

    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    feats.append(edges.mean())

    lap = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
    feats.append(lap.var())

    h, w = gray.shape
    q1 = norm[:h//2, :w//2].mean()
    q2 = norm[:h//2, w//2:].mean()
    q3 = norm[h//2:, :w//2].mean()
    q4 = norm[h//2:, w//2:].mean()
    feats += [abs(q1-q4), abs(q2-q3), np.std([q1, q2, q3, q4])]

    hist, _ = np.histogram(norm.ravel(), bins=16, range=(0, 1))
    feats += (hist / hist.sum()).tolist()

    cx, cy = w//4, h//4
    centre = norm[cy:3*cy, cx:3*cx].mean()
    feats.append(centre / (norm.mean() + 1e-6))

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    feats.append(np.sqrt(gx**2 + gy**2).mean())

    return np.array(feats, dtype=np.float32)


def load_dataset(data_dir: str):
    tumor_dir    = os.path.join(data_dir, "tumor")
    no_tumor_dir = os.path.join(data_dir, "no_tumor")

    X, y = [], []
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    for label, folder in [(1, tumor_dir), (0, no_tumor_dir)]:
        if not os.path.isdir(folder):
            print(f"[WARN] Folder not found: {folder}")
            continue
        files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]
        print(f"  {folder}: {len(files)} images (label={label})")
        for fname in files:
            path = os.path.join(folder, fname)
            try:
                img = Image.open(path)
                X.append(extract_features(img))
                y.append(label)
            except Exception as e:
                print(f"  [skip] {fname}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/", help="Root of dataset (tumor/ and no_tumor/ inside)")
    parser.add_argument("--output",   default="model.pkl", help="Where to save the trained model")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    print(f"\n📂  Loading dataset from: {args.data_dir}")
    X, y = load_dataset(args.data_dir)

    if len(X) == 0:
        print("[ERROR] No images found. Check your data_dir structure.")
        sys.exit(1)

    print(f"\n✅  Total samples: {len(X)}  (tumor={y.sum()}, no_tumor={(y==0).sum()})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.07,
            subsample=0.85, random_state=42
        ))
    ])

    print("\n🏋️  Training…")
    clf.fit(X_train, y_train)

    print("\n📊  Cross-validation (5-fold):")
    cv = cross_val_score(clf, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"   AUC = {cv.mean():.4f} ± {cv.std():.4f}")

    print("\n📋  Test set report:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["no_tumor", "tumor"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(clf, args.output)
    print(f"\n💾  Model saved → {args.output}")
    print("    Place model.pkl in the repo root, then run: streamlit run app.py")


if __name__ == "__main__":
    main()
