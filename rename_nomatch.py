import os

path = "E:/Han Project/TrainingDataset/ImagesPNG/"
path1 = "E:/Han Project/TrainingDataset/LabelsPNG/"

output = "E:/Han Project/TrainingDataset/RenamePNGImages/"
output1 = "E:/Han Project/TrainingDataset/RenamePNGLabels/"
r=0
# for i in os.listdir(path):
#     os.rename(path + i, output + str(r) + ".png")
#     r += 1
# r = 0
for j in os.listdir(path1):
    os.rename(path1 + j, output1 + str(r) + ".png")
    r += 1
