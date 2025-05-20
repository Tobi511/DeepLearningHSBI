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

# ---------------------------------------------------------------------------------------------------------
# Veränderung im Vergleich zu v9: initiale learning rate wieder etwas reduzieren
# ---------------------------------------------------------------------------------------------------------

version = "v10"
run = "r4"

# Ausgabe-Verzeichnisse und Dateien
results_dir = f"../results/results_{version}/results_{version}_{run}"
model_dir = f"../models/model_{version}"
model_path = os.path.join(model_dir, f"model_{version}_{run}.h5")
plot_path = os.path.join(results_dir, f"trainingsverlauf_{version}_{run}.png")
log_path = os.path.join(results_dir, f"trainingsverlauf_{version}_{run}.txt")
class_names_path = "../../../class_names.json"

# Pfad zu deinen Daten
train_data_dir = "../../../Vegetable Images/train"
val_data_dir = "../../../Vegetable Images/validation"

# Bildgröße und Batch-Größe
img_size = (32, 32)
batch_size = 128

# Trainings- und Validierungsdaten einlesen
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
with open(class_names_path, "w") as f:
    json.dump(class_names, f)

# Performance-Optimierung
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Modell definieren
model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),  # Eingabegröße

    layers.Conv2D(16, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),

    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),

    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),

    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),

    layers.Conv2D(256, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.15),

    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.Dropout(0.15),

    layers.Dense(15, activation='softmax')  # 15 Klassen
])

# Modell kompilieren
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

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

# Modell speichern
model.save(model_path)

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
plt.ylim(0, 1.2)  # Y-Achse fixieren
plt.xlim(left=0)
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
plt.yticks(ticks=[i * 0.2 for i in range(7)])
plt.legend(loc='lower right')
plt.title('Training und Validierungsgenauigkeit')

# Verlust
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.ylim(0, 2)  # Y-Achse fixieren
plt.xlim(left=0)
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
plt.yticks(ticks=[i * 0.2 for i in range(11)])
plt.legend(loc='upper right')
plt.title('Training und Validierungsverlust')

# Anzeige und Speicherung
os.makedirs(results_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(plot_path)

# Trainingsverlauf in Datei speichern
with open(log_path, "w") as f:
    f.write(f"{'Epoch':<6}{'Accuracy':<12}{'Val_Accuracy':<15}{'Loss':<12}{'Val_Loss':<12}\n")
    for epoch in epochs_range:
        f.write(
            f"{epoch + 1:<6}{acc[epoch]:<12.4f}{val_acc[epoch]:<15.4f}{loss[epoch]:<12.4f}{val_loss[epoch]:<12.4f}\n")
