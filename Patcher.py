import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from pyTiler.tiler import tiler

class Patcher:
    def __init__(self,
                 img,
                 lab=None,
                 batch_size=1,
                 input_shape=(512, 512, 1),
                 step=512,
                 num_classes=1,
                 overlap=0.2,
                 image=True,
                 visualize=False):
        self.img = img
        self.lab = lab
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.step = step
        self.num_classes = num_classes
        self.overlap = overlap
        self.image = image
        self.visualize = visualize


    def __getitem__(self, item):
        if self.lab is not None:
            return self.patch_image()[0][item], self.patch_image()[1][item]
        else:
            return self.patch_image()[0][item]

    def visualize_patch(self, patch_index):
        if 0 <= patch_index < self.num_patches:
            patch_img, patch_mask = self.__getitem__(patch_index)

            max_x_pixel, max_y_pixel = self.img.shape[:2]
            x_start = int((patch_index // self.patch_cols) * self.step * (1 - self.overlap))
            x_end = x_start + self.step
            y_start = int((patch_index % self.patch_cols) * self.step * (1 - self.overlap))
            y_end = y_start + self.step

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(self.img, cmap='gray')
            ax[0].add_patch(plt.Rectangle((y_start, x_start), self.step, self.step,
                                          edgecolor='r', facecolor='none', linewidth=2))
            ax[0].set_title('Original Image with Patch Outline')
            ax[1].imshow(patch_img.squeeze(), cmap='gray')
            ax[1].set_title('Patched Image')
            plt.show()
        else:
            print("Invalid patch index. Please choose a patch index between 0 and {}.".format(self.num_patches - 1))

    def patch_overlap(self, visualize=None):
        if visualize is not None:
            self.visualize = visualize
        else:
            visualize = False

        def normalize_image(input_block):
            block = input_block.copy()
            vol_max, vol_min = block.max(), block.min()
            if not vol_max == vol_min:  # run a check. otherwise error when divide by 0
                for i in range(block.shape[-1]):
                    block[:, :, i] = (block[:, :, i] - vol_min) / (vol_max - vol_min)
            return block

        max_x_pixel, max_y_pixel = self.img.shape[:2]
        # Calculate the number of patches to extract
        patch_rows = int(np.ceil(max_x_pixel / (self.step * (1 - self.overlap))))
        patch_cols = int(np.ceil(max_y_pixel / (self.step * (1 - self.overlap))))
        prev_x_start, prev_y_start = None, None
        num_patches = patch_rows * patch_cols
        # Create a numpy array to hold the patch data
        images = np.zeros((num_patches, self.step, self.step, self.num_classes), dtype=np.float32)
        if self.lab is not None:
            masks = np.zeros((num_patches, self.step, self.step, self.num_classes), dtype=bool)

        # Extract patches
        # # Pad the input image with zeros to handle patches going off the edge
        # padded_img = np.pad(self.img, ((0, self.step - 1), (0, self.step - 1), (0, 0)), mode='constant',
        #                     constant_values=0)
        # if self.lab is not None:
        #     padded_lab = np.pad(self.lab, ((0, self.step - 1), (0, self.step - 1), (0, 0)), mode='constant',
        #                         constant_values=0)

        # Extract patches
        patch_index = 0
        # Calculate the number of row and column patches
        num_row_patches = np.ceil(self.img.shape[0] / (self.step * (1 - self.overlap)))
        num_col_patches = np.ceil(self.img.shape[1] / (self.step * (1 - self.overlap)))

        # Compute the actual overlap for rows and columns
        actual_row_overlap = (num_row_patches * self.step - self.img.shape[0]) / ((num_row_patches - 1) * self.step)
        actual_col_overlap = (num_col_patches * self.step - self.img.shape[1]) / ((num_col_patches - 1) * self.step)

        # Use the adjusted overlap to generate the patches
        for row in range(int(num_row_patches)):
            for col in range(int(num_col_patches)):
                x_start = int(row * self.step * (1 - actual_row_overlap))
                x_end = x_start + self.step
                y_start = int(col * self.step * (1 - actual_col_overlap))
                y_end = y_start + self.step

                # Extract patch from the original image, possibly with smaller shape
                patch_img = self.img[x_start:min(x_end, max_x_pixel), y_start:min(y_end, max_y_pixel)]

                # Create an empty patch with the correct shape and fill it with the extracted data
                full_patch_img = np.zeros((self.step, self.step, self.num_classes), dtype=patch_img.dtype)
                full_patch_img[:patch_img.shape[0], :patch_img.shape[1], :] = patch_img[..., np.newaxis]

                patch_img = normalize_image(full_patch_img.reshape(self.step, self.step, self.num_classes))
                images[patch_index] = patch_img

                if self.lab is not None:
                    patch_mask = self.lab[x_start:min(x_end, max_x_pixel), y_start:min(y_end, max_y_pixel)]

                    # Create an empty mask with the correct shape and fill it with the extracted data
                    full_patch_mask = np.zeros((self.step, self.step, self.num_classes), dtype=bool)
                    if len(patch_mask.shape) == 2:  # if patch_mask is of shape (256,256)
                        patch_mask = patch_mask.reshape(patch_mask.shape[0], patch_mask.shape[1], 1)
                    full_patch_mask[:patch_mask.shape[0], :patch_mask.shape[1], :] = patch_mask

                    masks[patch_index] = full_patch_mask.reshape(self.step, self.step, self.num_classes)

                def exit_visualization(event):
                    self.visualize = False
                    plt.close('all')

                if self.visualize:  # Visualize the patch as it's being created
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                    ax[0].imshow(self.img, cmap='gray')
                    ax[0].add_patch(plt.Rectangle((y_start, x_start), self.step, self.step,
                                                  edgecolor='r', facecolor='none', linewidth=2))

                    if prev_x_start is not None and prev_y_start is not None:
                        ax[0].add_patch(plt.Rectangle((prev_y_start, prev_x_start), self.step, self.step,
                                                      edgecolor='b', facecolor='none', linewidth=2))

                    ax[0].set_title('Original Image with Patch Outline')
                    ax[1].imshow(patch_img.squeeze(), cmap='gray')
                    ax[1].set_title('Patched Image')

                    # Add exit button
                    button_ax = plt.axes([0.8, 0.01, 0.1, 0.05])  # x, y, width, height
                    exit_button = Button(button_ax, 'Exit')
                    exit_button.on_clicked(exit_visualization)

                    plt.show()

                    prev_x_start, prev_y_start = x_start, y_start
                # patch_img = normalize_image(patch_img.reshape(self.step, self.step, self.num_classes))
                # images[patch_index] = patch_img
                # if self.lab is not None:
                #     patch_mask = self.lab[x_start:x_end, y_start:y_end]
                #     patch_mask = np.array(patch_mask, dtype=bool)
                #     masks[patch_index] = patch_mask.reshape(self.step, self.step, self.num_classes)
                patch_index += 1
        if self.lab is not None:
            return images, masks, max_x_pixel, max_y_pixel
        else:
            return images, max_x_pixel, max_y_pixel

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
        if self.lab is not None:
            masks = np.zeros((self.batch_size, self.step, self.step, self.num_classes), dtype=bool)
        while cury < max_y_pixel:
            x_cur = curx
            while curx < max_x_pixel:

                # cropped = self.img.crop((curx, cury, curx+self.step, cury+self.step))
                # cropped_lab = self.lab.crop((curx, cury, curx+self.step, cury + self.step))
                cropped = self.img[curx:curx+self.step, cury:cury + self.step]
                if self.lab is not None:
                    cropped_lab = self.lab[curx:curx + self.step, cury:cury + self.step]
                    cropped_lab = np.array(cropped_lab, dtype=bool)

                curx = curx + self.step

                cropped = normalize_image(cropped.reshape(self.step, self.step, self.num_classes))

                images[i] = cropped # normalize_image(cropped.reshape(self.step, self.step, self.num_classes))
                if self.lab is not None:
                    masks[i] = cropped_lab.reshape(self.step, self.step, self.num_classes)
                i += 1
            curx = x_cur
            cury = cury + self.step
        if self.lab is not None:
            return images, masks, max_x_pixel, max_y_pixel
        else:
            return images, max_x_pixel, max_y_pixel

    def patch_image2(self):
        img = self.img
        n, m = self.input_shape[0], self.input_shape[1]
        overlap = 0

        t = tiler(img, n, m, overlap)
        tiles = t.tile(img, n, m, overlap)

        images = np.array(tiles).reshape((-1, n, m, self.num_classes))

        if self.lab is not None:
            t_lab = tiler(self.lab, n, m, overlap)
            tiles_lab = t_lab.tile(self.lab, n, m, overlap)
            masks = np.array(tiles_lab).reshape((-1, n, m, self.num_classes))

            return images, masks
        else:
            return images, max_x_tile, max_y_tile

    # def tilerPatch(self, img=self.img, overlap=0.2):
    #     t = tiler(img, img.shape[0], img.shape[1], overlap)
    #     out = t.tile(img, img.shape[0], img.shape[1], overlap)
    #     return out
