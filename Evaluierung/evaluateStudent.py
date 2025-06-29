import matplotlib
matplotlib.use('TkAgg')

import os
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import models, regularizers
import re
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
IMG_SIZE       = 224                  # MUSS zum Training passen!
BATCH_SIZE     = 128
NUM_CLASSES    = 15
WEIGHTS_PATH   = "../Meilenstein 4/Offline Distillation/models/model_KD_Opt_Teach/model_KD_Opt_Teach_r9/student_weights.h5"
TEST_DATA_DIR  = "..\\Vegetable Images\\test_v2"
# ────────────────────────────────────────────────────────────────────────────────


# Hier noch eine dynamische Anpassung für das Speichern hinzufügen
plot_path = ("../Meilenstein 4/Offline Distillation/results/results_KD_Opt_Teach/results_KD_Opt_Teach_r9/CM_r9_new_data_3")
# Pfad zur Textdatei
misclassified_path = "../Meilenstein 4/Offline Distillation/results/results_KD_Opt_Teach/results_KD_Opt_Teach_r9/misclassified_new_data_3.txt"


# 1) Test-Dataset laden und preprocess_input anwenden
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names  = test_ds.class_names

file_paths = []
for class_name in class_names:
    class_dir = os.path.join(TEST_DATA_DIR, class_name)
    images = sorted(os.listdir(class_dir))  # alphabetisch innerhalb der Klasse
    for img in images:
        file_paths.append(os.path.join(class_dir, img))

test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

# 2) Student-Modell definieren
def create_student_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(128, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(128, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling2D(2),
        layers.Conv2D(256, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling2D(2),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(5e-4)),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES)
    ])
    return model

model = create_student_model()

# 3) Gewichte laden
model.load_weights(WEIGHTS_PATH)

# 4) Vorhersagen & True Labels
y_pred_probs = model.predict(test_ds)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = np.concatenate([np.argmax(y, axis=1) for _, y in test_ds], axis=0)

#5) Confusion Matrix plotten
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

# 7) Verwechslungen analysieren und speichern
misclassified = []

for idx, (true, pred) in enumerate(zip(y_true, y_pred)):
    if true != pred:
        misclassified.append({
            "file": str(file_paths[idx]),
            "true_label": class_names[true],
            "predicted_label": class_names[pred]
        })

# 8) Ergebnisse speichern
print(f"Anzahl Verwechslungen: {len(misclassified)}")

# Schreiben im Format: Wahre_Klasse  Bildname  Vorhergesagt
with open(misclassified_path, "w") as f:
    for entry in misclassified:
        filename = os.path.basename(entry["file"])
        f.write(f"{entry['true_label']}\t{filename}\t{entry['predicted_label']}\n")
