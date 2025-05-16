import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from sklearn.model_selection import KFold
import pathlib

# Parameter
version = "v4_crossVali"
run = "r1"
k_folds = 5
img_size = (224, 224)
batch_size = 128
data_dir = pathlib.Path("../../Crossvalidation Vegetable Images")  # Beispielpfad
results_dir = f"../results/results_{version}/results_{version}_{run}"
model_dir = f"../models/model_{version}"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# 1) Bildpfade und Labels auslesen
all_image_paths = list(data_dir.glob("*/*.jpg"))  # je nachdem .jpg oder .png etc
all_image_paths = [str(path) for path in all_image_paths]
class_names = sorted({pathlib.Path(p).parent.name for p in all_image_paths})
class_to_index = dict((name, i) for i, name in enumerate(class_names))
all_labels = [class_to_index[pathlib.Path(p).parent.name] for p in all_image_paths]

# Klassennamen speichern
with open("../../class_names.json", "w") as f:
    json.dump(class_names, f)

all_labels = np.array(all_labels)

# 2) Funktion zum Laden & Vorverarbeiten der Bilder
def decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)  # oder decode_png
    image = tf.image.resize(image, img_size)
    image = image / 255.0
    return image, tf.one_hot(label, len(class_names))

# 3) Data Augmentation Layer
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# 4) K-Fold Split
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(all_image_paths)):
    print(f"\n--- Fold {fold + 1}/{k_folds} ---")

    train_paths = [all_image_paths[i] for i in train_idx]
    val_paths = [all_image_paths[i] for i in val_idx]
    train_labels = all_labels[train_idx]
    val_labels = all_labels[val_idx]

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

    # Bild laden + resize + onehot
    train_ds = train_ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)

    # Augmentation nur auf Training
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Modellaufbau exakt wie in deinem Single-Run
    model = keras.Sequential([
        layers.Input(shape=(*img_size, 3)),

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

        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=75,
        callbacks=[early_stop, lr_scheduler],
        verbose=2
    )

    # Speichern
    model_save_path = os.path.join(model_dir, f"model_{version}_{run}_fold{fold+1}.h5")
    model.save(model_save_path)

    # Plot erstellen und speichern (analog deinem Code)
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
    plt.xlim(left=0)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.yticks(ticks=[i * 0.2 for i in range(7)])
    plt.legend(loc='lower right')
    plt.title('Training und Validierungsgenauigkeit')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.ylim(0, 2)
    plt.xlim(left=0)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.yticks(ticks=[i * 0.2 for i in range(11)])
    plt.legend(loc='upper right')
    plt.title('Training und Validierungsverlust')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"trainingsverlauf_{version}_{run}_fold{fold+1}.png"))
    plt.close()

    # Logdatei schreiben
    with open(os.path.join(results_dir, f"trainingsverlauf_{version}_{run}_fold{fold+1}.txt"), "w") as f:
        f.write(f"{'Epoch':<6}{'Accuracy':<12}{'Val_Accuracy':<15}{'Loss':<12}{'Val_Loss':<12}\n")
        for epoch in epochs_range:
            f.write(f"{epoch+1:<6}{acc[epoch]:<12.4f}{val_acc[epoch]:<15.4f}{loss[epoch]:<12.4f}{val_loss[epoch]:<12.4f}\n")
