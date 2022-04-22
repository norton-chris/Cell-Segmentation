import numpy as np
import cv2

class Patcher:
    def __init__(self,
                 img,
                 batch_size = 1,
                 input_shape = (512, 512, 1),
                 step = 512,
                 num_classes = 1,
                 image = True):
        self.img = img
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.step = step
        self.num_classes = num_classes
        self.image = image

    def __getitem__(self, item):
        return self.patch_image()

    def patch_image(self):
        max_x_pixel = self.img.shape
        max_y_pixel = self.img.shape
        # batch_size = int((float(max_x_pixel[0]) / float(self.step)) * (float(max_y_pixel[0]) / float(self.step)))
        curx = 0
        cury = 0

        i = 0
        if self.image:
            images = np.empty((self.batch_size, self.step, self.step, self.num_classes))
        else:
            images = np.empty((self.batch_size, self.step, self.step, self.num_classes), dtype=bool)
        while cury < max_y_pixel[0]:
            x_cur = curx
            while curx < max_x_pixel[0]:

                cropped = self.img[curx:curx+self.step, cury:cury+self.step,:]


                curx = curx + self.step

                try:
                    images[i] = cropped.reshape(self.step, self.step, self.num_classes)
                    i += 1
                except:
                    pass
            curx = x_cur
            cury = cury + self.step

        return images, i

