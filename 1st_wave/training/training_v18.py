import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')

#---------------------------------------------------------------------------------------------------------
# # filters erhöhen, wegen höherer data augmentation, aber auch global average pooling
#---------------------------------------------------------------------------------------------------------

# Pfad zu deinen Daten
train_data_dir = "C:\\#Informatik Studium\\6. Semester\\DeepLearning\\Projekt stuff\\Vegetable Images\\train"
val_data_dir = "C:\\#Informatik Studium\\6. Semester\\DeepLearning\\Projekt stuff\\Vegetable Images\\validation"

# Bildgröße und Batch-Größe
img_size = (224, 224)
batch_size = 64  # 64 bzw 128 mal ausprobieren

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),  # ±36° statt nur ±18°
    layers.RandomZoom(0.1),  # etwas stärkerer Zoom
    layers.RandomTranslation(0.1, 0.1),  # Verschiebung in beide Richtungen
    layers.RandomContrast(0.1),  # Helligkeit/Kontrast leicht variieren
])

# Trainings- und Validierungsdaten einlesen (80/20 Split)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_data_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=True
)

# Klassen anzeigen
class_names = train_ds.class_names
print("Klassen:", class_names)

# Speichern der class_names
with open("../../class_names.json", "w") as f:
    json.dump(class_names, f)

# Data Augmentation anwenden NUR auf Trainingsdaten
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

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

    layers.Conv2D(512, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=2),

    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(15, activation='softmax')  # 15 Klassen
])

# Modell kompilieren
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

# Training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    batch_size=batch_size,
    callbacks=[early_stop, lr_scheduler]
)

# Modell speichern (optional)
model.save("models\\model_v18.h5")

# Trainingsverlauf plotten und speichern
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))

# Genauigkeit
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training und Validierungsgenauigkeit')

# Verlust
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training und Validierungsverlust')

# Anzeige und Speicherung
plt.tight_layout()
plt.savefig("results\\results_v18\\trainingsverlauf_v18.png")  # <- Speichert den Plot als Bild

# Ergebnisse in eine Textdatei speichern
results_dir = "../results/results_v18"
os.makedirs(results_dir, exist_ok=True)

with open(os.path.join(results_dir, "trainingsverlauf_v18.txt"), "w") as f:
    f.write("Epoch\tAccuracy\tVal_Accuracy\tLoss\tVal_Loss\n")
    for epoch in epochs_range:
        f.write(f"{epoch+1}\t{acc[epoch]:.4f}\t{val_acc[epoch]:.4f}\t{loss[epoch]:.4f}\t{val_loss[epoch]:.4f}\n")

