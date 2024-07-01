# import the necessary packages
import os

# path to training and testing data
DATASET = "dataset"
TRAIN_DATASET = "dataset/train"
TEST_DATASET = "dataset/test"

TRAIN_PRECENT = 0.8

#kaggle
KAGGLE_USERNAME = "moaaztarik"
KAGGLE_KEY = "692aa237541cbae7a86caed20a195c59"

# model input image size
IMAGE_SIZE = (128, 128)

# define the path to save the model
OUTPUT_PATH = "output"
MODEL_PATH = os.path.join(OUTPUT_PATH, "siamese_network")
OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_PATH, "output_image.png")