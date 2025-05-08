import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Prüfen, ob die GPU aktiv genutzt wird
print("Verfügbare GPUs:", tf.config.list_physical_devices('GPU'))

# Lade das MNIST-Datenset (handgeschriebene Ziffern)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Daten normalisieren (Werte 0–1 statt 0–255)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Eingabedaten für CNN um eine Kanal-Dimension erweitern
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Modell definieren
model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Kompilieren
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Modell trainieren
model.fit(x_train, y_train, epochs=3, validation_split=0.1)

# Modell evaluieren
model.evaluate(x_test, y_test)
