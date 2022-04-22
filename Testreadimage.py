import cv2
import matplotlib.pyplot as plt

train = "E:/Han Project/TrainingDataset/TrainFullDataset/Image_out/0.tif"

img = cv2.imread(train, -1)
print(img.shape)
plt.imshow(img)
plt.show()
cv2.imwrite("TESTIF16BIT.tif", img)