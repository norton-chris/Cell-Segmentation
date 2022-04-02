import os, sys
from PIL import Image
path = "E:/Han Project/TrainingDataset/TrainingDataset/Images/"
outpath = "E:/Han Project/TrainingDataset/TrainingDataset/ImagesPNG/"
error = 0
error_file = []
for infile in os.listdir(path):
    print("file : "+ path + infile)
    if infile[-3:] == "tif" or infile[-3:] == "TIF":
        outfile = infile[:-3] + "png"
        try:
            im = Image.open(path + infile)
            print("new filename: " + outfile)
            out = im.convert("RGB")
            out.save(outpath + outfile, "PNG", quality=100)
        except Exception as e:
            print(e)
            error += 1
            error_file.append(path + infile)



print("number of errors:", error)
print("image read error files: ")
for i in error_file:
    print(i)