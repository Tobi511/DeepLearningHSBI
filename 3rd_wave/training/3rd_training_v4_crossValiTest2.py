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
import numpy as np
from sklearn.model_selection import KFold

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# ---------------------------------------------------------------------------------------------------------
# K-Fold Cross-Validation
# ---------------------------------------------------------------------------------------------------------

version = "v4_CrossValiTest2"
run = "r1"
k_folds = 5

# Ausgabe-Verzeichnisse und Dateien
results_dir = f"../results/results_{version}/results_{version}_{run}"
model_dir = f"../models/model_{version}"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
class_names_path = "../../class_names.json"

# Pfad zu deinen Daten (alle gemeinsam in einem Verzeichnis)
data_dir = "..\\..\\Crossvalidation Vegetable Images"

# Bildgröße und Batch-Größe
img_size = (224, 224)
batch_size = 128

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Einlesen der Daten als Liste
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=img_size,
    label_mode="categorical",
    shuffle=True
)

class_names = dataset.class_names
with open(class_names_path, "w") as f:
    json.dump(class_names, f)

dataset = dataset.map(lambda x, y: (tf.keras.layers.Rescaling(1./255)(x), y))
all_images = []
all_labels = []
for images, labels in dataset:
    all_images.append(images)
    all_labels.append(labels)

#all_images = tf.concat(all_images, axis=0)
#all_labels = tf.concat(all_labels, axis=0)

all_images = np.concatenate([x.numpy() for x in all_images], axis=0)
all_labels = np.concatenate([y.numpy() for y in all_labels], axis=0)

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
    print(f"\n--- Fold {fold+1}/{k_folds} ---")

    #x_train, x_val = tf.gather(all_images, train_idx), tf.gather(all_images, val_idx)
    #y_train, y_val = tf.gather(all_labels, train_idx), tf.gather(all_labels, val_idx)

    x_train, x_val = all_images[train_idx], all_images[val_idx]
    y_train, y_val = all_labels[train_idx], all_labels[val_idx]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    train_ds = train_ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Modell definieren
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 3)),

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

        layers.Dense(15, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=75,
        callbacks=[early_stop, lr_scheduler],
        verbose=2
    )

    # Speichern
    model_path = os.path.join(model_dir, f"model_{version}_{run}_fold{fold+1}.h5")
    plot_path = os.path.join(results_dir, f"trainingsverlauf_{version}_{run}_fold{fold+1}.png")
    log_path = os.path.join(results_dir, f"trainingsverlauf_{version}_{run}_fold{fold+1}.txt")
    model.save(model_path)

    # Plot speichern
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
    plt.savefig(plot_path)
    plt.close()

    with open(log_path, "w") as f:
        f.write(f"{'Epoch':<6}{'Accuracy':<12}{'Val_Accuracy':<15}{'Loss':<12}{'Val_Loss':<12}\n")
        for epoch in epochs_range:
            f.write(
                f"{epoch + 1:<6}{acc[epoch]:<12.4f}{val_acc[epoch]:<15.4f}{loss[epoch]:<12.4f}{val_loss[epoch]:<12.4f}\n"
            )
