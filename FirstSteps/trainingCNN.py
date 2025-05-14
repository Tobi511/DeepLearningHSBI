import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
import os
import json

# Pfad zu deinen Daten
train_data_dir = "C:\\#Informatik Studium\\6. Semester\\DeepLearning\\Projekt stuff\\Vegetable Images\\train"
val_data_dir = "C:\\#Informatik Studium\\6. Semester\\DeepLearning\\Projekt stuff\\Vegetable Images\\validation"

# Bildgröße und Batch-Größe
img_size = (224, 224)
batch_size = 32  # 64 bzw 128 mal ausprobieren

# Trainings- und Validierungsdaten einlesen (80/20 Split)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical",
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_data_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical",
)

# Klassen anzeigen
class_names = train_ds.class_names
print("Klassen:", class_names)

# Speichern der class_names
with open("../class_names.json", "w") as f:
    json.dump(class_names, f)

# Performance-Optimierung
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Modell definieren
model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),  # Eingabegröße
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(256, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(15, activation='softmax')  # 15 Klassen
])

# Modell kompilieren
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    batch_size=batch_size,
    callbacks=[early_stop]
)

# Modell speichern (optional)
model.save("shot_Number_ONE.h5")
