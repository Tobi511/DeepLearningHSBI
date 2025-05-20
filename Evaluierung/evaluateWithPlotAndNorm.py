import matplotlib
matplotlib.use('TkAgg')

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model



# Testdaten einlesen
test_data_dir = "..\\Vegetable Images\\test"  # Passe den Pfad ggf. an

img_size = (224, 224)
batch_size = 128

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_data_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=False  # wichtig für korrekte Zuordnung
)

# Klassenliste merken
class_names = test_ds.class_names

normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Modell laden
model = load_model("../Meilenstein 2/3rd_wave/models/model_v1/model_v1_r1.h5")

# Vorhersagen generieren
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Wahre Labels extrahieren
y_true = np.concatenate([np.argmax(label, axis=1) for _, label in test_ds], axis=0)

# Confusion Matrix berechnen
cm = confusion_matrix(y_true, y_pred)

# Plotten
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Vorhergesagt")
plt.ylabel("Tatsächlich")
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Optional: Klassifikationsbericht
print(classification_report(y_true, y_pred, target_names=class_names))
