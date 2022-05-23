import os
import shutil

img_in_path = "TrainingDataset/output/Images/"
img_out_path = "TrainingDataset/correct_labels_subset/Images/"
lab_in_path = "TrainingDataset/correct_labels_subset/Labels/"

for i in os.listdir(lab_in_path):
    s = i.split(".")
    print(img_in_path + s[0] + ".tif")
    shutil.copy(img_in_path + s[0] + ".tif", img_out_path + s[0] + ".tif")

