import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
import os

# === Parameter ===
image_path = "C:\\#Informatik Studium\\6. Semester\\DeepLearning\\Projekt stuff\\Vegetable Images\\test\\Potato\\1005.jpg"  # Pfad zum Bild anpassen
model_path = "../1st_wave/models/model_v1.h5"  # Modellpfad
class_names_path = "../class_names.json"  # JSON mit Klassen

img_size = (224, 224)

# === Modell und Klassen laden ===
model = load_model(model_path)

with open(class_names_path, "r") as f:
    class_names = json.load(f)

# === Bild laden und vorbereiten ===
img = Image.open(image_path).convert("RGB")
img = img.resize(img_size)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, axis=0)  # Batch-Dimension hinzufügen
img_array = img_array / 255.0  # Optional: Normalisierung

# === Vorhersage ===
pred = model.predict(img_array)
predicted_index = np.argmax(pred, axis=1)[0]
predicted_class = class_names[predicted_index]
confidence = pred[0][predicted_index] * 100

# === Ausgabe ===
print(f"Vorhergesagte Klasse: {predicted_class} ({confidence:.2f}%)")

plt.imshow(img)
plt.title(f"Vorhergesagt: {predicted_class} ({confidence:.2f})")
plt.axis('off')
plt.show()