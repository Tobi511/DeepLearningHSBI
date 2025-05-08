import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers
import os

# Pfad zu deinen Daten
data_dir = "C:\\Users\\Tobi\\Desktop\\Vegetable Images\\train"

# Bildgröße und Batch-Größe
img_size = (224, 224)
batch_size = 32 #64 bzw 128 mal ausprobieren

# Trainings- und Validierungsdaten einlesen (80/20 Split)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# Klassen anzeigen
class_names = train_ds.class_names
print("Klassen:", class_names)

# Performance-Optimierung
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Modell definieren
model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),  # Normalisierung
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')  # Anzahl Klassen
])

# Modell kompilieren
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(patience=3, restore_best_weights=True)

# Training
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)

# Modell speichern (optional)
model.save("mein_cnn_klassifikator.h5")
