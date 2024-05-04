# import the necessary packages
import os

# path to training and testing data
DATASET = "dataset"
TRAIN_DATASET = "dataset/train"
TEST_DATASET = "dataset/test"


# model input image size
IMAGE_SIZE = (128, 128)


# batch size and the buffer size
BATCH_SIZE = 128
BUFFER_SIZE = BATCH_SIZE * 2

# define the training parameters
LEARNING_RATE = 0.0001
STEPS_PER_EPOCH = 50
VALIDATION_STEPS = 10
EPOCHS = 10

# define the path to save the model
OUTPUT_PATH = "output"
MODEL_PATH = os.path.join(OUTPUT_PATH, "siamese_network")
OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_PATH, "output_image.png")