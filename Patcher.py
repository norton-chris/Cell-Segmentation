import numpy as np
import cv2

class Patcher:
    def __init__(self,
                 img,
                 lab,
                 batch_size = 1,
                 input_shape = (512, 512, 1),
                 step = 512,
                 num_classes = 1,
                 image = True):
        self.img = img
        self.lab = lab
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.step = step
        self.num_classes = num_classes
        self.image = image

    def __getitem__(self, item):
        return self.patch_image()

    def patch_image(self):
        def normalize_image(input_block):
            block = input_block.copy()
            vol_max, vol_min = block.max(), block.min()
            if not vol_max == vol_min:  # run a check. otherwise error when divide by 0
                for i in range(block.shape[-1]):
                    block[:, :, i] = (block[:, :, i] - vol_min) / (vol_max - vol_min)
            return block
        max_x_pixel = self.img.shape[0]
        max_y_pixel = self.img.shape[1]
        # batch_size = int((float(max_x_pixel[0]) / float(self.step)) * (float(max_y_pixel[0]) / float(self.step)))
        curx = 0
        cury = 0

        i = 0
        images = np.zeros((self.batch_size, self.step, self.step, self.num_classes), dtype="float32")
        masks = np.zeros((self.batch_size, self.step, self.step, self.num_classes), dtype=bool)
        while cury < max_y_pixel:
            x_cur = curx
            while curx < max_x_pixel:

                # cropped = self.img.crop((curx, cury, curx+self.step, cury+self.step))
                # cropped_lab = self.lab.crop((curx, cury, curx+self.step, cury + self.step))
                cropped = self.img[curx:curx+self.step, cury:cury + self.step]
                cropped_lab = self.lab[curx:curx + self.step, cury:cury + self.step]
                cropped_lab = np.array(cropped_lab, dtype=bool)

                curx = curx + self.step

                cropped = normalize_image(cropped.reshape(self.step, self.step, self.num_classes))

                images[i] = cropped # normalize_image(cropped.reshape(self.step, self.step, self.num_classes))
                masks[i] = cropped_lab.reshape(self.step, self.step, self.num_classes)
                i += 1
            curx = x_cur
            cury = cury + self.step

        return images, masks, max_x_pixel, max_y_pixel

