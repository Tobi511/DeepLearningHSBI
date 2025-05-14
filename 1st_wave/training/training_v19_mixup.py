import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import json
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')

# ---------------------------------------------------------------------------------------------------------
# MixUp-Funktionen
# ---------------------------------------------------------------------------------------------------------

def sample_beta_distribution(size, alpha):
    return np.random.beta(alpha, alpha, size)

def mixup(batch_x, batch_y, alpha=0.2):
    lam = sample_beta_distribution(batch_x.shape[0], alpha)
    lam_x = lam.reshape(-1, 1, 1, 1).astype(np.float32)
    lam_y = lam.reshape(-1, 1).astype(np.float32)

    index = tf.random.shuffle(tf.range(batch_x.shape[0]))
    x_shuffled = tf.gather(batch_x, index)
    y_shuffled = tf.gather(batch_y, index)

    x_mixed = lam_x * batch_x + (1 - lam_x) * x_shuffled
    y_mixed = lam_y * batch_y + (1 - lam_y) * y_shuffled

    return x_mixed, y_mixed

def apply_mixup(batch_x, batch_y):
    x_mixed, y_mixed = tf.numpy_function(
        func=mixup,
        inp=[batch_x, batch_y],
        Tout=[tf.float32, tf.float32]
    )
    x_mixed.set_shape(batch_x.shape)
    y_mixed.set_shape(batch_y.shape)
    return x_mixed, y_mixed

# ---------------------------------------------------------------------------------------------------------
# Datenpfade und Bildkonfiguration
# ---------------------------------------------------------------------------------------------------------

train_data_dir = "C:\\#Informatik Studium\\6. Semester\\DeepLearning\\Projekt stuff\\Vegetable Images\\train"
val_data_dir = "C:\\#Informatik Studium\\6. Semester\\DeepLearning\\Projekt stuff\\Vegetable Images\\validation"

img_size = (224, 224)
batch_size = 64

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
])

# ---------------------------------------------------------------------------------------------------------
# Dataset laden
# ---------------------------------------------------------------------------------------------------------

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

# Klassen speichern
class_names = train_ds.class_names
print("Klassen:", class_names)

with open("../../class_names.json", "w") as f:
    json.dump(class_names, f)

# Daten augmentieren + MixUp
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.map(apply_mixup)

# Performance-Optimierung
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ---------------------------------------------------------------------------------------------------------
# Modell definieren
# ---------------------------------------------------------------------------------------------------------

model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),

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

    layers.Flatten(),

    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(15, activation='softmax')
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

# ---------------------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    batch_size=batch_size,
    callbacks=[early_stop, lr_scheduler]
)

# Modell speichern
model.save("models\\model_v19_mixup.h5")

# ---------------------------------------------------------------------------------------------------------
# Ergebnisse plotten und speichern
# ---------------------------------------------------------------------------------------------------------

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training und Validierungsgenauigkeit')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training und Validierungsverlust')

plt.tight_layout()
os.makedirs("../results/results_v19_mixup", exist_ok=True)
plt.savefig("results\\results_v19_mixup\\trainingsverlauf_v19_mixup.png")

# Werte in Textdatei schreiben
with open("results\\results_v19_mixup\\trainingsverlauf_v19_mixup.txt", "w") as f:
    f.write("Epoch\tAccuracy\tVal_Accuracy\tLoss\tVal_Loss\n")
    for epoch in epochs_range:
        f.write(f"{epoch+1}\t{acc[epoch]:.4f}\t{val_acc[epoch]:.4f}\t{loss[epoch]:.4f}\t{val_loss[epoch]:.4f}\n")
