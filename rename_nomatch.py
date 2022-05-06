import os
import shutil


path = "TrainingDataset/Images/"
path1 = "TrainingDataset/Labels/"

output = "TrainingDataset/Images512/"
output1 = "TrainingDataset/Labels512/"
r=0
for i in os.listdir(path):
    try:
        shutil.copy(path + i, output + str(r) + ".tif")
    except IOError as io_err:
        os.makedirs(os.path.dirname(output))
        shutil.copy(path + i, output + str(r) + ".tif")
        pass

    #shutil.copy(path + i, output + str(r) + ".tif")
    #os.rename(path + i, output + str(r) + ".png")
    r += 1
r = 0
for j in os.listdir(path1):
    try:
        shutil.copy(path1 + i, output1 + str(r) + ".tif")
    except IOError as io_err:
        os.makedirs(os.path.dirname(output1))
        shutil.copy(path1 + i, output1 + str(r) + ".tif")
        pass
    #os.rename(path1 + j, output1 + str(r) + ".png")
    r += 1
