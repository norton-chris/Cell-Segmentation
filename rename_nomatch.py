import os
import shutil


path = "E:/Han Project/TrainingDataset/clean_dataset/TrainingDataset/Images/"
path1 = "E:/Han Project/TrainingDataset/clean_dataset/TrainingDataset/Labels/"

output = "E:/Han Project/TrainingDataset/clean_dataset/TrainingDataset/RenamedImages/"
output1 = "E:/Han Project/TrainingDataset/clean_dataset/TrainingDataset/RenamedLabels/"
r=0
for i in os.listdir(path):
    shutil.copy(path + i, output + str(r) + ".png")
    #os.rename(path + i, output + str(r) + ".png")
    r += 1
r = 0
for j in os.listdir(path1):
    shutil.copy(path1 + i, output1 + str(r) + ".png")
    #os.rename(path1 + j, output1 + str(r) + ".png")
    r += 1
