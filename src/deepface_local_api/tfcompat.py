import os

# TensorFlow 2.16+ uses Keras 3 unless this is set before tensorflow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
