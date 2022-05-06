import cv2
import os

train  = "TrainingDataset/Cropped_images/"
labels = "TrainingDataset/Cropped_labels/"

too_small = 0
for i in os.listdir(train):
    img = cv2.imread(train + i)
    #print("img shape:", img.shape[0])
    if(img.shape[0] < 512 or img.shape[1] < 512):
        os.remove(train + i)
        os.remove(labels + i)
        too_small += 1
    print(i)
    print(img.shape)

print("too small:", too_small)