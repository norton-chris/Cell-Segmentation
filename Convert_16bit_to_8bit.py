from PIL import Image
import os
import numpy as np

PATH = "TrainingDataset/output/Images/"

for i in os.listdir(PATH):
    img = Image.fromarray(np.array(Image.open(PATH + i)).astype("uint8"))
    train_img = img.save("TrainingDataset/uint8TIF/" + i)