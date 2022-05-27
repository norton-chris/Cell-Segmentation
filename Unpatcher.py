import numpy as np
import cv2
from PIL import Image


class Unpatcher:
    def __init__(self,
                 img,
                 patches,
                 img_name,
                 batch_size = 1,
                 input_shape = (512, 512, 1),
                 step = 512,
                 num_classes = 1,
                 image = True):
        self.img = img
        self.patches = patches
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.step = step
        self.num_classes = num_classes
        self.image = image
        self.img_name = img_name

    def __getitem__(self, item):
        return self.unpatch_image()

    def unpatch_image(self):
        img = Image.fromarray(np.array(Image.open(self.img_name)))
        max_x_pixel = img.size[0]
        max_y_pixel = img.size[1]
        step = self.patches.shape[1]
        # batch_size = int((float(max_x_pixel[0]) / float(self.step)) * (float(max_y_pixel[0]) / float(self.step)))
        curx = 0
        cury = 0

        i = 0
        #images = np.empty((self.batch_size, self.step, self.step, self.num_classes))
        if len(img.size) == 2:
            masks = np.zeros((img.size[0], img.size[1]), dtype=bool)
            xstep = self.patches.shape[1]
            ystep = self.patches.shape[2]
        elif len(img.size) == 3:
            masks = np.zeros((img.size[0], img.size[1], img.size[2]), dtype=bool)
            xstep = self.patches.shape[1]
            ystep = self.patches.shape[2]

        new_img_array = np.zeros(shape=(max_x_pixel, max_y_pixel, 1))
        while cury < max_y_pixel:
            #x_cur = curx
            #x_offset = 0

            #curx = curx + xstep
            while curx < max_x_pixel:
                # img = Image.open(self.patches)
                # new_im.paste(img, (x_offset, 0))
                # x_offset += img.size[curx, 0]
                #img.paste(self.patches[i], (curx, cury, curx + xstep, cury + ystep))
                new_img_array[curx:curx + xstep, cury:cury + ystep] = self.patches[i]
                #cropped_lab = np.array(cropped_lab, dtype=bool)

                curx = curx + xstep

                i += 1

            #masks = np.vstack(self.patches[i])
            #i += 1

            curx = 0
            cury = cury + ystep

        return new_img_array

