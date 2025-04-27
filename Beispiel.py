import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. Lade den MNIST-Datensatz (60000 Trainingsbilder und 10000 Testbilder)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 2. Normalisiere die Eingabedaten (Bilder) auf einen Bereich von [0, 1]
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0

# 3. Konvertiere die Labels in One-Hot-Encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 4. Erstelle das Modell (ein einfaches CNN)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 Klassen für die Ziffern 0 bis 9
])

# 5. Kompiliere das Modell
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Trainiere das Modell
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

# 7. Teste das Modell
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Test accuracy: {test_acc}")
