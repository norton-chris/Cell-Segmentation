import cv2
import os
import matplotlib.pyplot as plt

path = "TrainingDataset/data_subset/output/test/Images/"

for i in os.listdir(path):
    # read image
    img = cv2.imread(path + i)

    # convert to grey
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # perform Rosin thres

    # perform Otsu thres
    otsu = cv2.threshold(img, 0, 122, cv2.THRESH_OTSU)


    # make matplotlib figure

    fig = plt.figure(figsize=(20, 14))

    # create subplot
    #fig.add_subplot(1, 2, 1)
    # showing image
    #plt.imshow(rosin)
    #plt.axis('off')
    #plt.title("rosin")
    # create subplot
    fig.add_subplot(1, 1, 1)
    # showing image
    plt.imshow(otsu)
    plt.axis('off')
    plt.title("otsu")

    # show plot
    plt.show()