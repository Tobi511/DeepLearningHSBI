import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import json

# Testdaten vorbereiten
test_data_dir = "C:\\#Informatik Studium\\6. Semester\\DeepLearning\\Projekt stuff\\Vegetable Images\\test"
img_size = (224, 224)
batch_size = 32

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_data_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=False
)

# Klassen laden
with open("../../class_names.json", "r") as f:
    class_names = json.load(f)

# Modell laden
model = load_model("../../Meilenstein 2/1st_wave/models/model_v1.h5")

# Einzelnes Bild aus dem Dataset nehmen
for img_batch, label_batch in test_ds.take(1):  # Nur das erste Batch
    for i in range(len(img_batch)):  # Alle Bilder im Batch durchgehen
        img = img_batch[i]
        label = label_batch[i]

        # Vorhersage
        img_expanded = tf.expand_dims(img, axis=0)  # (1, 224, 224, 3)
        pred = model.predict(img_expanded, verbose=0)

        # Ausgabe
        true_label = class_names[np.argmax(label.numpy())]
        predicted_label = class_names[np.argmax(pred)]

        print(f"Bild {i + 1}:")
        print(f" → Tatsächliche Klasse : {true_label}")
        print(f" → Vorhergesagte Klasse: {predicted_label}")
        print("---")

