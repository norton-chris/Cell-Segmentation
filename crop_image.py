import numpy as np
import cv2
import os

PATH = "TrainingDataset/Labels/"
TRAIN_PATH = "TrainingDataset/Images/"
for i in os.listdir(PATH):
    img = cv2.imread(PATH + i)

    #cv2.imshow("img", img)
    #cv2.waitKey(0)
    xmin = img.shape[0]
    ymin = img.shape[1]
    xmax = 0
    ymax = 0
    for y in range(0,img.shape[1]):
        for x in range(0, img.shape[0]):
            if all(img[y,x] == [255,255,255]):
                if xmin > x:
                    xmin = x
                if ymin > y:
                    ymin = y
                if xmax < x:
                    xmax = x
                if ymax < y:
                    ymax = y
    
    #print("xmax:", xmax)
    #print("xmin:", xmax)
    #print("ymax:", ymax)
    #print("ymin:", ymin)
    # add 20 px margin
    xmax += 20
    ymax += 20
    xmin -= 20
    ymin -= 20
    #img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),2)
    img = img[ymin:ymax, xmin:xmax]
    cv2.imwrite("TrainingDataset/Cropped/Labels/", img)
    train_img = cv2.imread(TRAIN_PATH + i)
    train_img = train_img[ymin:ymax, xmin:xmax]
    cv2.imwrite("TrainingDataset/Cropped/Images/", train_img)
    print(PATH + i)
    #cv2.imshow("image", img)
    #cv2.waitKey(0)

            
