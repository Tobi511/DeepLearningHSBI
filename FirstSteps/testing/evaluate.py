import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# Testdaten einlesen
test_data_dir = "C:\\#Informatik Studium\\6. Semester\\DeepLearning\\Projekt stuff\\Vegetable Images\\test"  # Pfad anpassen, falls nötig

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_data_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical"
)

# Modell laden
model = load_model("FirstSteps\\shot_Number_ONE.h5")

# Testen
loss, accuracy = model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
