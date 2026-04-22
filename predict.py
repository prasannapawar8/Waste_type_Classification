import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # ✅ Fix preprocessing
import sys
import os

model       = tf.keras.models.load_model("best_model.h5")
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def predict(img_path):
    # ✅ Validate file exists
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        return

    img  = keras_image.load_img(img_path, target_size=(224, 224))
    arr  = preprocess_input(keras_image.img_to_array(img))  # ✅ Fixed: was /255.0 (wrong for MobileNetV2)
    inp  = np.expand_dims(arr, axis=0)
    preds = model.predict(inp, verbose=0)[0]

    print("\n🗑️  Waste Classification Results:")
    print("─" * 40)
    for name, prob in sorted(zip(CLASS_NAMES, preds), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 30)
        print(f"{name:12s} {bar:30s} {prob:.2%}")
    print(f"\n✅ Prediction: {CLASS_NAMES[np.argmax(preds)].upper()}")
    print("─" * 40)

# ✅ Interactive loop — keeps asking until user types 'exit'
while True:
    print("\n📂 Enter image path (or 'exit' to quit):")
    img_path = input(">>> ").strip().strip('"')  # strip quotes if user drags file in

    if img_path.lower() == 'exit':
        print("👋 Goodbye!")
        break

    predict(img_path)