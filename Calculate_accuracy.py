import os
import cv2
import numpy as np

test = "TrainingDataset/output/test/"
label = cv2.imread(test + 'Capture1.jpg',0) # labeled image
prediction = cv2.imread('/path/to/Capture1.jpg',0) # prediction image

def calculate_accuracy(label, prediction):
    rows,cols = label.shape
    total_pixels = rows * cols
    correct_pixels = 0
    for i in range(rows):
        for j in range(cols):
            lab_pix = label[i,j]
            pred_pix = prediction[i,j]
            if lab_pix != pred_pix:
                continue
            correct_pixels += 1

    #print("Accuracy:", correct_pixels / total_pixels)
    return correct_pixels / total_pixels

