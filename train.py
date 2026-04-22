import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# ─── CONFIG ───────────────────────────────────────────────
DATASET_PATH = "trashnet_data/dataset-resized"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS       = 30        # ✅ Increased from 25
SEED         = 42

# ─── DATA GENERATORS ──────────────────────────────────────
# ✅ Added more aggressive augmentation to reduce overfitting
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,          # ✅ Increased from 20
    zoom_range=0.3,             # ✅ Increased from 0.2
    width_shift_range=0.2,      # ✅ Increased from 0.1
    height_shift_range=0.2,     # ✅ Increased from 0.1
    horizontal_flip=True,
    vertical_flip=True,         # ✅ NEW
    brightness_range=[0.7, 1.3],# ✅ NEW - helps with plastic/glass lighting confusion
    shear_range=0.2,            # ✅ NEW
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=SEED
)

val_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,              # ✅ Important for evaluation
    seed=SEED
)

print("✅ Classes found:", train_gen.class_indices)
NUM_CLASSES = len(train_gen.class_indices)

# ✅ NEW — Compute class weights to fix trash/cardboard underperformance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weight_dict = dict(enumerate(class_weights))
print("📊 Class weights:", {list(train_gen.class_indices.keys())[i]: round(v, 2) 
                             for i, v in class_weight_dict.items()})

# ─── IMPROVED MODEL ───────────────────────────────────────
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu')(x)   # ✅ Increased from 256
x = layers.Dropout(0.5)(x)                    # ✅ Increased from 0.4
x = layers.Dense(256, activation='relu')(x)   # ✅ NEW second dense layer
x = layers.Dropout(0.3)(x)                    # ✅ NEW
output = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─── CALLBACKS ────────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True, verbose=1),  # ✅ patience 5→7
    ModelCheckpoint("best_model.h5", save_best_only=True, verbose=1),
    ReduceLROnPlateau(factor=0.3, patience=3, verbose=1, min_lr=1e-7)  # ✅ factor 0.5→0.3
]

# ─── PHASE 1: Train top layers only ───────────────────────
print("\n🔒 Phase 1: Training top layers (base frozen)...")
history1 = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen,
    callbacks=callbacks,
    class_weight=class_weight_dict   # ✅ NEW
)

# ─── PHASE 2: Fine-tune last 50 layers ────────────────────
print("\n🔓 Phase 2: Fine-tuning last 50 layers...")  # ✅ increased from 30
base_model.trainable = True
for layer in base_model.layers[:-50]:               # ✅ unfreeze more layers
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),  # ✅ lower LR: 1e-4 → 5e-5
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    class_weight=class_weight_dict   # ✅ NEW
)

# ─── PLOT TRAINING CURVES ─────────────────────────────────
def plot_history(h1, h2):
    acc   = h1.history['accuracy']     + h2.history['accuracy']
    val   = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss  = h1.history['loss']         + h2.history['loss']
    vloss = h1.history['val_loss']     + h2.history['val_loss']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(acc, label='Train Accuracy')
    axes[0].plot(val, label='Val Accuracy')
    axes[0].axvline(x=len(h1.history['accuracy'])-1,
                    color='red', linestyle='--', label='Fine-tune start')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)   # ✅ NEW

    axes[1].plot(loss,  label='Train Loss')
    axes[1].plot(vloss, label='Val Loss')
    axes[1].axvline(x=len(h1.history['loss'])-1,
                    color='red', linestyle='--', label='Fine-tune start')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)   # ✅ NEW

    plt.suptitle('MobileNetV2 — Waste Classification Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("✅ Saved training_curves.png")

plot_history(history1, history2)

# ✅ NEW — Print final validation accuracy
final_val_acc = max(history1.history['val_accuracy'] + history2.history['val_accuracy'])
print(f"\n🏆 Best Validation Accuracy: {final_val_acc:.2%}")
print("✅ Training complete! Model saved as best_model.h5")