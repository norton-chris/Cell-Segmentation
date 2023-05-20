import numpy as np
import cv2
from PIL import Image

from pyTiler.tiler import tiler

class Unpatcher:
    def __init__(self,
                 img,
                 patches,
                 img_name,
                 batch_size = 1,
                 input_shape = (512, 512, 1),
                 step = 512,
                 num_classes = 1,
                 image = True,
                 overlap=0.2):
        self.img = img
        self.patches = patches
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.step = step
        self.num_classes = num_classes
        self.image = image
        self.img_name = img_name
        self.overlap = overlap

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

    # def unpatch_image2(self):
    #     pytiler = tiler(self.img, self.img.shape[0], self.img.shape[1], overlap=self.overlap)
    #     full_img = pytiler.untile(self.patches)
    #     return full_img
    def unpatch_image2(self):
        max_x_pixel, max_y_pixel = self.img.shape[:2]

        patch_rows = int(np.ceil(max_x_pixel / (self.step * (1 - self.overlap))))
        patch_cols = int(np.ceil(max_y_pixel / (self.step * (1 - self.overlap))))

        full_pred_image = np.zeros((max_x_pixel, max_y_pixel), dtype=np.float32)

        patch_index = 0
        for row in range(patch_rows):
            for col in range(patch_cols):
                x_start = int(row * self.step * (1 - self.overlap))
                x_end = x_start + self.step
                y_start = int(col * self.step * (1 - self.overlap))
                y_end = y_start + self.step

                patch_pred = self.patches[patch_index].squeeze()

                # Determine the actual patch size without padding
                actual_x_size = min(x_end, max_x_pixel) - x_start
                actual_y_size = min(y_end, max_y_pixel) - y_start

                # Update full_pred_image with patch_pred without padding
                if actual_x_size < self.step or actual_y_size < self.step:
                    full_pred_image[x_start:x_start + actual_x_size, y_start:y_start + actual_y_size] = patch_pred[
                                                                                                        :actual_x_size,
                                                                                                        :actual_y_size]
                else:
                    full_pred_image[x_start:x_end, y_start:y_end] = patch_pred

                patch_index += 1

        return full_pred_image


