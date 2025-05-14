import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
import os
import json

# Lade das Modell
model = load_model("../1st_wave/models/model_v1.h5")

# Klassen definieren (müssen identisch zur Reihenfolge im Training sein!)
#class_names = [
#    'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
#    'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
#    'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'
#]

with open("../class_names.json", "r") as f:
    class_names = json.load(f)

# Bildpfad eingeben
image_path = "C:\\#Informatik Studium\\6. Semester\\DeepLearning\\Projekt stuff\\Vegetable Images\\test\\Potato\\1005.jpg"  # <<< HIER ANPASSEN

# Bild vorbereiten
img = load_img(image_path, target_size=(224, 224))  # gleiche Größe wie beim Training
img_array = img_to_array(img)
# img_array = img_array / 255.0  # Normalisieren (wenn kein Rescaling-Layer im Modell)
img_array = np.expand_dims(img_array, axis=0)  # Batch-Dimension hinzufügen

# Vorhersage
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]
confidence = predictions[0][predicted_index] * 100

# Ausgabe
print(f"Bild: {os.path.basename(image_path)}")
print(f"Vorhergesagte Klasse: {predicted_class} ({confidence:.2f} % Sicherheit)")
