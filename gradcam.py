import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ─── LOAD MODEL ───────────────────────────────────────────
model = tf.keras.models.load_model("best_model.h5")

CLASS_NAMES     = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
LAST_CONV_LAYER = "out_relu"

# ─── GRAD-CAM FUNCTION ────────────────────────────────────
def get_gradcam(model, img_array, layer_name):
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        pred_idx = tf.argmax(preds[0])
        class_score = preds[:, pred_idx]

    grads    = tape.gradient(class_score, conv_out)
    pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap  = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap  = tf.squeeze(heatmap).numpy()
    heatmap  = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)

    return heatmap, CLASS_NAMES[pred_idx], float(tf.reduce_max(preds))

# ─── OVERLAY FUNCTION ─────────────────────────────────────
def overlay_gradcam(img_path):
    # Load original image for display (0-255, RGB)
    img_display = keras_image.load_img(img_path, target_size=(224, 224))
    img_display = keras_image.img_to_array(img_display).astype(np.uint8)  # Keep 0-255 for display

    # Preprocessed input for model (-1 to +1)
    arr = preprocess_input(img_display.copy().astype(np.float32))
    inp = np.expand_dims(arr, axis=0)

    # Get Grad-CAM
    heatmap, pred_class, confidence = get_gradcam(model, inp, LAST_CONV_LAYER)

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (224, 224))

    # Colorize heatmap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on original image
    superimposed = cv2.addWeighted(img_display, 0.6, heatmap_color, 0.4, 0)

    # ─── PLOT ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(img_display)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap", fontsize=13)
    axes[1].axis('off')

    axes[2].imshow(superimposed)
    axes[2].set_title(f"Overlay\nPred: {pred_class.upper()}\nConfidence: {confidence:.2%}", fontsize=13)
    axes[2].axis('off')

    plt.suptitle("Grad-CAM Visualization — Waste Classification", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig("gradcam_output.png", dpi=150)
    plt.show()
    print(f"✅ Predicted: {pred_class.upper()} | Confidence: {confidence:.2%}")
    print("✅ Saved gradcam_output.png")

# ─── TEST WITH MULTIPLE IMAGES ────────────────────────────
test_images = [
    "trashnet_data/dataset-resized/plastic/plastic1.jpg",
    "trashnet_data/dataset-resized/glass/glass1.jpg",
    "trashnet_data/dataset-resized/metal/metal1.jpg",
]

for img_path in test_images:
    print(f"\n📸 Processing: {img_path}")
    overlay_gradcam(img_path)