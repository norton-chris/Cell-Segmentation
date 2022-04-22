import os

path = "E:/Han Project/TrainingDataset/clean_dataset/TrainingDataset/Images/"
path1 = "E:/Han Project/TrainingDataset/clean_dataset/TrainingDataset/Labels/"

output = "E:/Han Project/TrainingDataset/clean_dataset/TrainingDataset/RenamedImages"
output1 = "E:/Han Project/TrainingDataset/clean_dataset/TrainingDataset/RenamedLabels"
r=0
found = False
total = 0
err_total = 0
for i in os.listdir(path):
	for j in os.listdir(path1):
		if j == i:
			os.rename(path1 + j, output1 + str(r) + ".png")
			found = True
			break
	if not found:
		print("no match found. Image file name:", i)
		err_total += 1
	else:
		os.rename(path + i, output + str(r) + ".png")
		r += 1
	found = False
	total += 1

print(str(err_total) + "/" + str(total))
