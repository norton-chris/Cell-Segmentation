import numpy as np
import cv2
import os
from PIL import Image
import time
import psutil

PATH = "TrainingDataset/output/Labels/"
TRAIN_PATH = "TrainingDataset/output/Images/"
errors = 0
for i in os.listdir(PATH):
    print(PATH + i)
    img = Image.fromarray(np.array(Image.open(PATH + i)).astype("uint16"))
    pixel = img.load()
    try:

        #cv2.imshow("img", img)
        #cv2.waitKey(0)
        xmin = img.size[0]
        ymin = img.size[1]
        xmax = 0
        ymax = 0

        for y in range(0,img.size[1]):
            for x in range(0, img.size[0]):
                white = pixel[y, x] == 1
                if white:
                    #pixel[y,x] = 255
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

        #img = img[ymin:ymax, xmin:xmax]
        img = img.crop((xmin, ymin, xmax, ymax))
        # print(xmax, ymax, xmin, ymin)
        # print(img.size)
        # img.show()
        # time.sleep(5)
        # for proc in psutil.process_iter():
        #     if proc.name() == "display":
        #         proc.kill()
        img = img.save("TrainingDataset/Cropped_labels/" + i)
        train_img = Image.fromarray(np.array(Image.open(TRAIN_PATH + i)).astype("uint16"))
        #train_img = train_img[ymin:ymax, xmin:xmax]
        train_img = train_img.crop((xmin, ymin, xmax, ymax))
        train_img = train_img.save("TrainingDataset/Cropped_images/" + i)
    except Exception as e:
        print(e)
        print("error on file:", i)
        print("current # of errors:", errors)
        errors += 1
        pass
    #cv2.imshow("image", img)
    #cv2.waitKey(0)

print("errors:", errors)
            
