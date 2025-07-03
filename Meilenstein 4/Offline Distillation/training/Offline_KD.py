import tensorflow as tf
from tensorflow.keras import layers, regularizers, optimizers, losses, callbacks, models, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os
import json
from matplotlib import pyplot as plt
import numpy as np

#
# r1 - T=5 a=1.0
# r2 - T=5 a=0.5 <--- Auswahl für nächste Reihe
# r3 - T=5 a=0.2

# r4 - T=8 a=0.5  |
# r5 - T=2 a=0.5  | keine Verbesserung zu r2
# r6 - T=3 a=0.5  |
# r7 - T=3 a=0.3  |

# r8 - T=5 a=0.8
# r9 - T=5 a=0.3 <--- Bisher bestes Ergebnis

# -------------------- Paths and Parameters --------------------
version = "KD_Opt_Teach"
run = "r9"
TEMPERATURE = 5
ALPHA = 0.3

results_dir = f"../results/results_{version}/results_{version}_{run}"
model_dir = f"../models/model_{version}/model_{version}_{run}"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

train_data_dir = "../../../Vegetable Images/train"
val_data_dir = "../../../Vegetable Images/validation"
log_path = os.path.join(results_dir, f"transfertrainingsverlauf_{version}_{run}.txt")
class_names_path = os.path.join(model_dir, "class_names.json")
WEIGHTS_PATH = "../storage_old/model_ResNet50_tuning/weights_ResNet50_tuning_r5.h5"

IMG_SIZE = 224
BATCH_SIZE = 128
NUM_CLASSES = 15
EPOCHS_TEACHER = 20
EPOCHS_DISTILL = 20

# -------------------- Data Pipeline --------------------
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

# Save class names
class_names = train_ds.class_names
with open(class_names_path, "w") as f:
    json.dump(class_names, f)

# Augmentation on training only
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

train_ds = train_ds.map(lambda x, y: (preprocess_input(data_augmentation(x, training=True)), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

# Performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -------------------- Teacher Model --------------------
teacher_backbone = ResNet50(
    include_top=False,
    weights=None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

tf.keras.backend.clear_session()

# Partial Fine-Tuning: nur letzte 18 Layer trainierbar
teacher_backbone.trainable = True  # True
for layer in teacher_backbone.layers[:-18]:
    layer.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = teacher_backbone(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(NUM_CLASSES)(x)

teacher_model = Model(inputs, outputs)

teacher_model.load_weights(WEIGHTS_PATH)

# -------------------- Student Model --------------------
def create_student_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(128, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(128, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling2D(2),
        layers.Conv2D(256, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling2D(2),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(5e-4)),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES)  #softmax entfernt, logits
    ])
    return model

student_model = create_student_model()

# -------------------- Distiller Class --------------------
class Distiller(tf.keras.Model):
    def __init__(self, student, teacher, temperature=3.0, alpha=0.7):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss_fn = losses.CategoricalCrossentropy(from_logits=True)
        self.kld_loss_fn = losses.KLDivergence()

    def compile(self, optimizer, metrics):
        super().compile(optimizer=optimizer, metrics=metrics)

    # Override train_step für Loss Berechnung in der Distillation
    def train_step(self, data):
        x, y_true = data
        teacher_logits = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)
            soft_teacher = tf.nn.softmax(teacher_logits / self.temperature)
            soft_student = tf.nn.softmax(student_logits / self.temperature)
            soft_loss = self.kld_loss_fn(soft_teacher, soft_student) * (self.temperature ** 2)
            hard_loss = self.ce_loss_fn(y_true, student_logits)
            loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.compiled_metrics.update_state(y_true, student_logits)
        results = {m.name: m.result() for m in self.metrics}
        results['combined_loss'] = loss
        results['hard_loss'] = hard_loss
        results['soft_loss'] = soft_loss
        return results

    # Override test_step für Evaluation, normaler Val_loss für Vergleichbarkeit
    def test_step(self, data):
        x, y_true = data
        student_logits = self.student(x, training=False)
        loss = self.ce_loss_fn(y_true, student_logits)
        self.compiled_metrics.update_state(y_true, student_logits)
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss
        return results

    def call(self, inputs):
        return self.student(inputs)

# -------------------- Distillation Training --------------------
# Freeze teacher
teacher_model.trainable = False

distiller = Distiller(student=student_model, teacher=teacher_model,
                      temperature=TEMPERATURE, alpha=ALPHA)
distiller.compile(
    optimizer=optimizers.Adam(1e-4),
    metrics=["accuracy"]
)

reduce_lr_s = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

history_distill = distiller.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_DISTILL,
    callbacks=[reduce_lr_s]
)

# Save student weights
distiller.student.save_weights(os.path.join(model_dir, "student_weights.h5"))

# -------------------- Visualization --------------------
acc = history_distill.history['accuracy']
val_acc = history_distill.history['val_accuracy']
combined_loss = history_distill.history['combined_loss']
hard_loss = history_distill.history['hard_loss']
soft_loss = history_distill.history['soft_loss']
val_loss = history_distill.history['val_loss']
epochs_range = range(1, len(acc) + 1)

# 1. Accuracy Plot
plt.figure(figsize=(12, 4))
plt.plot(epochs_range, acc, label='Train Acc')
plt.plot(epochs_range, val_acc, label='Val Acc')
plt.ylim(0, 1)
plt.xlim(left=0)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.title('Distillation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "accuracy_plot.png"))

# 2. Hard Loss + Validation Loss
plt.figure(figsize=(12, 4))
plt.plot(epochs_range, hard_loss, label='Hard Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.ylim(0, 1)
plt.xlim(left=0)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.title('Hard Loss & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "hard_val_loss_plot.png"))

# 3. Combined Loss + Soft Loss separat
plt.figure(figsize=(12, 4))
plt.plot(epochs_range, combined_loss, label='Combined Loss')
plt.plot(epochs_range, soft_loss, label='Soft Loss')
plt.ylim(0, max(max(combined_loss), max(soft_loss)) * 1.1)
plt.xlim(left=0)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.title('Soft Loss & Combined Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "soft_combined_loss_plot.png"))

with open(log_path, "w") as f:
    f.write(f"{'Epoch':<6}{'Acc':<14}{'Val_Acc':<14}{'Combined_Loss':<14}{'Val_Loss':<14}{'Hard_Loss':<14}{'Soft_Loss':<14}\n")
    for i in epochs_range:
        f.write(
            f"{i:<6}"
            f"{acc[i - 1]:<14.6f}"
            f"{val_acc[i - 1]:<14.6f}"
            f"{combined_loss[i - 1]:<14.6f}"
            f"{val_loss[i - 1]:<14.6f}"
            f"{hard_loss[i - 1]:<14.6f}"
            f"{soft_loss[i - 1]:<14.6f}\n"
        )

