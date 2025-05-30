import tensorflow as tf
from tensorflow.keras.applications import (
    MobileNetV2,
    EfficientNetB0,
    EfficientNetB3,
    ResNet50
)

# Einheitliche Input-Shape
IMG_SIZE = 224
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Modelle mit include_top=False laden (ohne Klassifikationskopf)
MODELS = {
    "MobileNetV2": MobileNetV2,
    "EfficientNetB0": EfficientNetB0,
    "EfficientNetB3": EfficientNetB3,
    "ResNet50": ResNet50
}

print(f"{'Modell':<20} {'# Layers':<10} {'10% Layers':<12}")
print("-" * 42)

for name, constructor in MODELS.items():
    model = constructor(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE
    )
    num_layers = len(model.layers)
    ten_percent = int(round(num_layers * 0.1))

    print(f"{name:<20} {num_layers:<10} {ten_percent:<12}")
