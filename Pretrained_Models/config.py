"""
Configuration file for DeepFake Detection Project
Contains all the constants and configuration variables
"""

import os

# Dataset paths (using existing archive dataset)
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "archive", "real_vs_fake", "real-vs-fake")
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'valid')
TEST_DIR = os.path.join(BASE_DIR, 'test')

# Image dimensions (reduced for faster training)
IMG_WIDTH = 128  # Reduced from 256 to 128
IMG_HEIGHT = 128  # Reduced from 256 to 128

# Training parameters (optimized for CPU)
BATCH_SIZE = 16
EPOCHS = 3  # Reduced from 10 to 3 for quick training
LEARNING_RATE = 1e-4

# Image dimensions (reduced for faster training)
IMG_WIDTH = 128  # Reduced from 256 to 128
IMG_HEIGHT = 128  # Reduced from 256 to 128

# Model save paths
MODEL_SAVE_DIR = "saved_models/"
VGG16_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "fine_tuned_vgg16_last5.h5")
VGG19_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "fine_tuned_vgg19_last5.h5")
INCEPTION_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "fine_tuned_inception_last5.h5")
RESNET50_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "fine_tuned_resnet50_last5.h5")

# Create model save directory if it doesn't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
