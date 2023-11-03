import os
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import Scoring

# Define the normalization function
def normalize_image(input_block):
    block = input_block.copy()
    vol_max, vol_min = block.max(), block.min()
    if vol_max != vol_min:
        block = (block - vol_min) / (vol_max - vol_min)
    return block

root = "TrainingDataset/data_subset/323_subset/output/train/"
test = root + "Images/"

# Load the model
model = load_model('h5_files/unet_multi512shp16Flt0.0lr1000E_2imgs-20231103-08:04.h5',
                   custom_objects={'dice_plus_bce_loss': Scoring.dice_plus_bce_loss,
                                   'dice_scoring': Scoring.dice_scoring})

# Iterate over test images
for filename in os.listdir(test):
    img_path = test + filename
    label_path = root + "Labels/" + filename

    # Read and normalize the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype("float32")
    img_normalized = normalize_image(img)

    # Read the label
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

    # Reshape for the model (add batch dimension)
    img_normalized = img_normalized[np.newaxis, ..., np.newaxis]

    # Predict
    preds = model.predict(img_normalized)

    # Post-process predictions if necessary (e.g., thresholding)
    # preds = (preds > threshold).astype(np.uint8)

    # Visualize the prediction
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(label, cmap='gray')
    plt.title('Label')
    plt.axis('off')

    # Define a confidence threshold
    confidence_threshold = 0.01  # This can be any value between 0 and 1

    # Apply the threshold to the predictions
    preds_binary = (preds > confidence_threshold).astype(np.uint8)

    plt.subplot(1, 3, 3)
    plt.imshow(preds_binary[0, ..., 0], cmap='gray')  # Assuming single-channel output
    plt.title('Prediction')
    plt.axis('off')

    plt.show()