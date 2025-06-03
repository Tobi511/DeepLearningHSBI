import matplotlib
matplotlib.use('TkAgg')

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
IMG_SIZE       = 224                  # MUSS zum Training passen!
BATCH_SIZE     = 128
NUM_CLASSES    = 15
WEIGHTS_PATH   = "../Meilenstein 3/Fine Tuning/models/model_ResNet50_tuning/weights_ResNet50_tuning_r4.h5"
TEST_DATA_DIR  = "..\\Vegetable Images\\test"
# ────────────────────────────────────────────────────────────────────────────────

# Hier noch eine dynamische Anpassung für das Speichern hinzufügen
plot_path = ("../Meilenstein 3/Fine Tuning/results/results_ResNet50_tuning/results_ResNet50_tuning_r4"
             "/Confusionmatrix_ResNet50_tuning_r4.png")


# 1) Test-Dataset laden und preprocess_input anwenden
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)
class_names  = test_ds.class_names

test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

# 2) Architektur exakt nach Training rekonstruieren
backbone = ResNet50(
    include_top=False,
    weights=None,                  # NICHT 'imagenet'!
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
# Partial Fine-Tuning: nur letzte 20 Layer trainierbar
backbone.trainable = True  # True
for layer in backbone.layers[:-18]:
    layer.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = backbone(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
#x = layers.Dense(256, activation="relu")(x)
#x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

# 3) Gewichte laden
model.load_weights(WEIGHTS_PATH)

# 4) Vorhersagen & True Labels
y_pred_probs = model.predict(test_ds)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = np.concatenate([np.argmax(y, axis=1) for _, y in test_ds], axis=0)

# 5) Confusion Matrix plotten
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Vorhergesagt")
plt.ylabel("Tatsächlich")
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(plot_path)

# 6) Klassifikationsbericht
print(classification_report(y_true, y_pred, target_names=class_names))
