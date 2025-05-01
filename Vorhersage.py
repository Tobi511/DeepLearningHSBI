import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model("mein_cnn_klassifikator.h5")
# Bild laden und vorbereiten
img_path = "C:\\Users\\Tobi\\Desktop\\Vegetable Images\\test\\Radish\\1011.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Batch-Dimension hinzufügen
img_array = img_array / 255.0  # Normalisieren

# Vorhersage
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])

# Klassenliste muss bekannt sein (wie im Training)
class_names = ['Bean','Bitter_Gourd','Bottle_Gourd','Brinjal','Broccoli','Cabbage','Capsicum','Carrot','Cauliflower','Cucumber','Papaya','Potato','Pumpkin','Radish','Tomato']  # Deine 10 Klassen
print(f"Vorhergesagte Klasse: {class_names[predicted_class]} ({confidence*100:.2f}% Sicherheit)")
