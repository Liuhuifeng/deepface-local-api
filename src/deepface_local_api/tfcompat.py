import os

# TensorFlow 2.16+ uses Keras 3 unless this is set before tensorflow is imported.
# DeepFace still expects tf.keras (Keras 2) APIs.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
