import os
import cv2
import numpy as np

test = "TrainingDataset/output/test/"
label = cv2.imread(test + 'Capture1.jpg',0) # labeled image
prediction = cv2.imread('/path/to/Capture1.jpg',0) # prediction image
rows,cols,_ = label.shape
total_pixels = rows * cols
correct_pixel = 0
for i in range(rows):
    for j in range(cols):
        lab_pix = label[i,j]
        pred_pix = prediction[i,j]
        if lab_pix != pred_pix:
            continue
        correct_pixel += 1

print("Accuracy:", correct_pixel / total_pixels)

