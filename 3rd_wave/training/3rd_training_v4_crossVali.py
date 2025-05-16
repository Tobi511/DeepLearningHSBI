import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from tensorflow.keras import layers
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

tf.get_logger().setLevel('ERROR')

# ---------------------------------------------------------------------------------------------------------------------
version = "v4_crossVali"
run = "r1"
k_folds = 5
batch_size = 128
img_size = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE

# Pfade
base_data_dir = "..\\..\\Crossvalidation Vegetable Images"
results_dir = f"../results/results_{version}/results_{version}_{run}"
model_dir = f"../models/model_{version}"
class_names_path = "../../class_names.json"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Initiale Einlesung für Klassennamen
full_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_data_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=True
)
class_names = full_ds.class_names
with open(class_names_path, "w") as f:
    json.dump(class_names, f)

# Liste aller Dateipfade und Labels extrahieren
file_paths = []
labels = []

for file_batch, label_batch in full_ds.unbatch():
    file_paths.append(file_batch)
    labels.append(label_batch)

file_paths = np.array(file_paths)
labels = np.array(labels)

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(file_paths)):
    print(f"\n--- Fold {fold + 1}/{k_folds} ---")

    x_train = tf.gather(file_paths, train_idx)
    y_train = tf.gather(labels, train_idx)
    x_val = tf.gather(file_paths, val_idx)
    y_val = tf.gather(labels, val_idx)

    # Tensorflow-Datensätze erzeugen
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    # Augmentation und Normalisierung
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])
    normalization = tf.keras.layers.Rescaling(1. / 255)

    train_ds = train_ds.map(lambda x, y: (normalization(data_augmentation(x, training=True)), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    train_ds = train_ds.shuffle(1024).batch(batch_size).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    # Modell definieren
    model = tf.keras.Sequential([
        layers.Input(shape=(224, 224, 3)),  # Eingabegröße

        layers.Conv2D(32, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(64, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(128, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Conv2D(128, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Conv2D(128, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(256, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=2),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(15, activation='softmax')  # 15 Klassen
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=75,
        callbacks=[early_stop, lr_scheduler],
        verbose=2
    )

    # Speichern
    model_path = os.path.join(model_dir, f"model_{version}_{run}_fold{fold + 1}.h5")
    model.save(model_path)

    # Plotten
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.ylim(0, 1.2)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(loc='lower right')
    plt.title('Training und Validierungsgenauigkeit')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.ylim(0, 2)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(loc='upper right')
    plt.title('Training und Validierungsverlust')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"trainingsverlauf_{version}_{run}_fold{fold + 1}.png"))
    plt.close()

    # Trainingsverlauf speichern
    with open(os.path.join(results_dir, f"trainingsverlauf_{version}_{run}_fold{fold + 1}.txt"), "w") as f:
        f.write(f"{'Epoch':<6}{'Accuracy':<12}{'Val_Accuracy':<15}{'Loss':<12}{'Val_Loss':<12}\n")
        for epoch in epochs_range:
            f.write(
                f"{epoch + 1:<6}{acc[epoch]:<12.4f}{val_acc[epoch]:<15.4f}{loss[epoch]:<12.4f}{val_loss[epoch]:<12.4f}\n"
            )
