import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json

from matplotlib import gridspec
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

# ----------------------------- Konfiguration -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 15

TEST_DATA_DIR = "../../../Vegetable Images/test_v2"  # <- Anpassen
MODEL_WEIGHTS_PATH = "../models/model_KD_Opt_Teach/model_KD_Opt_Teach_r9/student_weights.h5"  # <- Anpassen
CLASS_NAMES_JSON = "../models/model_KD_Opt_Teach/model_KD_Opt_Teach_r9/class_names.json"  # <- Anpassen
SAVE_DIR = "./gradcam_output"  # Ausgabeordner

# Nur falsch klassifizierte speichern / alle speichern
ONLY_MISCLASSIFIED = False


# ----------------------------- Modell laden -----------------------------
def load_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(128, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(128, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(256, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(5e-4)),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES)
    ])
    return model


# ----------------------------- Grad-CAM Utilities -----------------------------
def get_gradcam_heatmap(model, image, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()


def save_gradcam(image, heatmap, save_path, alpha=0.4, true_label=None, pred_label=None):
    # Konvertiere PIL zu RGB und dann zu NumPy
    image = image.convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, img_bgr, 1 - alpha, 0)

    # Matplotlib-Visualisierung mit Colorbar & Labels
    fig = plt.figure(figsize=(8, 6))
    spec = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=[20, 1], height_ratios=[20, 1], wspace=0.05, hspace=0.15)

    ax_img = fig.add_subplot(spec[0, 0])
    ax_img.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    ax_img.axis('off')

    ax_cbar = fig.add_subplot(spec[0, 1])
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('Aktivierung', rotation=270, labelpad=15)

    ax_label = fig.add_subplot(spec[1, 0])
    ax_label.axis('off')
    label_text = f"True: {true_label}    |    Pred: {pred_label}"
    ax_label.text(0.5, 0.5, label_text, ha='center', va='center', fontsize=11)

    # Speichern
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# ----------------------------- Hauptskript -----------------------------
if __name__ == "__main__":
    with open(CLASS_NAMES_JSON, "r") as f:
        class_names = json.load(f)

    model = load_model()
    model.load_weights(MODEL_WEIGHTS_PATH)

    # Datensatz laden
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DATA_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        label_mode="categorical",
        shuffle=False
    )
    class_names_ds = test_ds.class_names
    file_paths = []

    # Originale Dateipfade abrufen
    for class_name in class_names_ds:
        class_dir = os.path.join(TEST_DATA_DIR, class_name)
        for fname in sorted(os.listdir(class_dir)):
            file_paths.append(os.path.join(class_dir, fname))

    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

    y_true = []
    y_pred = []

    for idx, (image_batch, label_batch) in enumerate(test_ds):
        image_path = file_paths[idx]
        image_pil = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE))
        label_true_idx = np.argmax(label_batch.numpy()[0])
        pred_logits = model(image_batch, training=False)
        label_pred_idx = tf.argmax(pred_logits[0]).numpy()

        y_true.append(label_true_idx)
        y_pred.append(label_pred_idx)

        if ONLY_MISCLASSIFIED and label_true_idx == label_pred_idx:
            continue  # nur Fehler speichern

        heatmap = get_gradcam_heatmap(model, image_batch, last_conv_layer_name="conv2d_5")
        fname = os.path.basename(image_path)
        save_path = os.path.join(SAVE_DIR, class_names[label_true_idx], fname)
        save_gradcam(image_pil, heatmap, save_path, true_label=class_names[label_true_idx], pred_label=class_names[label_pred_idx])

        print(f"[{idx+1}/{len(file_paths)}] Saved: {save_path}")

    print("Grad-CAM Generierung abgeschlossen.")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
