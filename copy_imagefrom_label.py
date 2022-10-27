#----------------------------------------------------------------------------
# Created By  : Chris Norton
# ---------------------------------------------------------------------------
"""
When creating new labels I needed to grab matching images
and put them in corresponding folders.
"""
# ---------------------------------------------------------------------------

import os
import shutil

img_in_path = "../TrainingDataset/Renamed_PNG/"
img_out_path = "TrainingDataset/data_subset/jpg_label_images/Images/"
lab_in_path = "TrainingDataset/data_subset/jpg_label_images/Labels/"

for i in os.listdir(lab_in_path):
    s = i.split(".")
    print(img_in_path + s[0] + ".tif")
    shutil.copy(img_in_path + s[0] + ".tif", img_out_path + s[0] + ".tif")

