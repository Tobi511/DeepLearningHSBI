import tensorflow as tf

# Prüfe, ob TensorFlow GPUs erkennt
print("Verfügbare GPUs:", tf.config.list_physical_devices('GPU'))
