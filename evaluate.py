import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ─── CONFIG ───────────────────────────────────────────────
DATASET_PATH = "trashnet_data/dataset-resized"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32

# ─── LOAD MODEL ───────────────────────────────────────────
model = tf.keras.models.load_model("best_model.h5")
print("✅ Model loaded")

# ─── DATA ─────────────────────────────────────────────────
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_names = list(val_gen.class_indices.keys())

# ─── PREDICTIONS ──────────────────────────────────────────
y_pred_probs = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_gen.classes

# ─── CLASSIFICATION REPORT ────────────────────────────────
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ─── CONFUSION MATRIX ─────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap='Blues')
plt.title("Confusion Matrix — Waste Classification")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("✅ Saved confusion_matrix.png")