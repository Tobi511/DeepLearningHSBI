import json
import os
import tensorflow as tf
from matplotlib import pyplot as plt

# Nur tensorflow.keras verwenden
from tensorflow.keras import layers, regularizers, Model, optimizers, losses, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# --------------------------------------------------------------------------------------------------------------------


# EfficientNetB0 - Feature Extraction


# -------------------------------------------------------------------------------------------------------------------


tf.get_logger().setLevel('ERROR')

version = "EfficientNetB0_FE"
run = "r1"

# Pfade anlegen
results_dir = f"../results/results_{version}/results_{version}_{run}"
model_dir = f"../models/model_{version}"
model_path = os.path.join(model_dir, f"model_{version}_{run}")
plot_path = os.path.join(results_dir, f"transfertrainingsverlauf_{version}_{run}.png")
log_path = os.path.join(results_dir, f"transfertrainingsverlauf_{version}_{run}.txt")
class_names_path = "../../../class_names.json"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Datenpfade
train_data_dir = "../../../Vegetable Images/train"
val_data_dir = "../../../Vegetable Images/validation"

# Parameter
IMG_SIZE = 224
BATCH_SIZE = 128
NUM_CLASSES = 15
EPOCHS = 20

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),  # ±36° statt nur ±18°
    layers.RandomZoom(0.1),  # etwas stärkerer Zoom
    layers.RandomContrast(0.1),  # Helligkeit/Kontrast leicht variieren
])

# Dataset laden (preprocess_input direkt in Directory-Loader)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=123,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_data_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Klassen speichern
class_names = train_ds.class_names
with open(class_names_path, "w") as f:
    json.dump(class_names, f)


# Data Augmentation anwenden NUR auf Trainingsdaten
train_ds = train_ds.map(lambda x, y: (preprocess_input(data_augmentation(x, training=True)), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

# Performance-Optimierung
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# EfficientNetB3 Backbone laden
backbone = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
# Partial Fine-Tuning: nur letzte 20 Layer trainierbar
backbone.trainable = False  # True
# for layer in backbone.layers[:-20]:
# layer.trainable = False

# Functional-API Modellaufbau
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = backbone(inputs, training=False)  # Backbone im Inferenz-Modus für BatchNorm
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

# Kompilieren
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.summary()

# Early Stopping
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,       # halbiert die Lernrate
    patience=3,
    min_lr=1e-6
)

# Training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[es, reduce_lr]
)

# Modell im TensorFlow-SavedModel-Format speichern
# model.save(model_path, save_format="tf")

weights_path = os.path.join(model_dir, f"weights_{version}_{run}.h5")
model.save_weights(weights_path)

# Trainingsverlauf plotten und speichern
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Acc')
plt.plot(epochs_range, val_acc, label='Val   Acc')
plt.ylim(0, 1.2)  # Y-Achse fixieren
plt.xlim(left=0)
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
plt.yticks(ticks=[i * 0.2 for i in range(7)])
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val   Loss')
plt.ylim(0, 2)  # Y-Achse fixieren
plt.xlim(left=0)
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
plt.yticks(ticks=[i * 0.2 for i in range(11)])
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(plot_path)

# Trainingslog abspeichern
with open(log_path, "w") as f:
    f.write(f"{'Epoch':<6}{'Acc':<10}{'Val_Acc':<10}{'Loss':<10}{'Val_Loss':<10}\n")
    for i in epochs_range:
        f.write(f"{i:<6}{acc[i - 1]:<10.4f}{val_acc[i - 1]:<10.4f}{loss[i - 1]:<10.4f}{val_loss[i - 1]:<10.4f}\n")
