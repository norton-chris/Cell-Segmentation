import os
path = "TrainingDataset/data_subset/Labels/"

for i in range(1,600):
    if not os.path.isfile(path + str(i) + ".tif"):
        print(str(i) + ".tif does not exist")