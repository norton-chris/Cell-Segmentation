import os
import shutil


path = "E:/Han Project/TrainingDataset/clean_dataset/Images/"
path1 = "E:/Han Project/TrainingDataset/clean_dataset/Labels/"

output = "E:/Han Project/TrainingDataset/clean_dataset/TrainingDataset/Images/"
output1 = "E:/Han Project/TrainingDataset/clean_dataset/TrainingDataset/Labels/"
r=0
for i in os.listdir(path):
    shutil.copy(path + i, output + str(r) + ".tif")
    #os.rename(path + i, output + str(r) + ".png")
    r += 1
r = 0
for j in os.listdir(path1):
    shutil.copy(path1 + j, output1 + str(r) + ".tif")
    #os.rename(path1 + j, output1 + str(r) + ".png")
    r += 1
