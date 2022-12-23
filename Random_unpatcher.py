import numpy as np
import cv2
from PIL import Image
import random

class Random_unpatcher:
    def __init__(self,
                 img,
                 img_name,
                 model,
                 batch_size = 1,
                 input_shape = (512, 512, 1),
                 step = 512,
                 num_classes = 1,
                 image = True,
                 num_crop = 25):
        self.img = img
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.step = step
        self.num_classes = num_classes
        self.image = image
        self.img_name = img_name
        self.model = model
        self.num_crop = num_crop

    def __getitem__(self, item):
        return self.unpatch_image()

    def random_unpatch(self):
        # Load the image
        image = cv2.imread(self.img_name, -1)

        # Determine the size of the crops
        crop_size = (self.input_shape[0], self.input_shape[1])

        # Initialize an empty prediction map
        prediction_map = np.zeros(image.shape[:2], dtype=np.float32)
        predicted_pixels = np.zeros(image.shape[:2], dtype=np.bool)

        # Number of crops to take
        num_crops = self.num_crop

        # Iterate over the number of crops
        for _ in range(num_crops):
            # Randomly select the top-left corner of the crop
            x = random.randint(0, image.shape[0] - crop_size[0])
            y = random.randint(0, image.shape[1] - crop_size[1])
            print("patch coords:", x, y)

            # Crop the image
            crop = image[x:x + crop_size[0], y:y + crop_size[1]]

            # Resize the crop to the input size of the model
            crop = cv2.resize(crop, (self.model.input_shape[1], self.model.input_shape[2]))

            # Normalize the crop
            crop = (crop / 255.0) - 0.5

            # Add the batch dimension
            crop = np.expand_dims(crop, axis=0)
            crop = crop.reshape(1, self.step, self.step, self.num_classes)

            # Perform inference on the crop
            prediction = self.model.predict(crop)


            # Update the prediction map with the prediction for this crop
            prediction = prediction.reshape(self.step, self.step)
            prediction_map[x:x + crop_size[0], y:y + crop_size[1]] = prediction
            predicted_pixels[x:x + crop_size[0], y:y + crop_size[1]] = True

        unpredicted_pixels = np.logical_not(predicted_pixels)

        # Iterate over the image in crops
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if unpredicted_pixels[i, j]:
                    print("missed crop..", i, j, "creating new patch..")
                    # Crop the image
                    crop = image[i:i + crop_size[0], j:j + crop_size[1]]

                    # Resize the crop to the input size of the model
                    crop = cv2.resize(crop, (self.model.input_shape[1], self.model.input_shape[2]))

                    # Normalize the crop
                    crop = (crop / 255.0) - 0.5

                    # Add the batch dimension
                    crop = np.expand_dims(crop, axis=0)
                    crop = crop.reshape(1, self.step, self.step, self.num_classes)

                    # Perform inference on the crop
                    prediction = self.model.predict(crop)

                    # Update the prediction map with the prediction for this crop
                    prediction = prediction.reshape(self.step, self.step)
                    prediction_map[i:i + crop_size[0], j:j + crop_size[1]] = prediction
                    predicted_pixels[x:x + crop_size[0], y:y + crop_size[1]] = True
                    unpredicted_pixels = np.logical_not(predicted_pixels)
        # Normalize the prediction map
        prediction_map = (prediction_map - prediction_map.min()) / (prediction_map.max() - prediction_map.min())

        # Save the prediction map
        cv2.imwrite('prediction_map.jpg', prediction_map * 255)

        return prediction_map

    def efficient_random_unpatch(self):
        # Load the image
        image = cv2.imread(self.img_name, -1)

        # Determine the size of the crops
        crop_size = (self.input_shape[0], self.input_shape[1])

        prediction_map = np.zeros(image.shape[:2], dtype=np.float32)
        unpredicted_mask = np.ones(image.shape[:2], dtype=np.bool)

        # Number of crops to take
        num_crops = self.num_crop

        # Iterate over the number of crops
        i = 0
        for _ in range(num_crops):
            if not np.any(unpredicted_mask):
                print("all pixels have been predicted in", i, "iterations.. breaking..")
                break
            # Select a random crop from the unpredicted pixels
            x, y = np.where(unpredicted_mask)
            x = x[random.randint(0, len(x) - 1)]
            y = y[random.randint(0, len(y) - 1)]
            print("patch:", x, y)
            x = x - (x % crop_size[0])
            y = y - (y % crop_size[1])

            # Crop the image
            crop = image[x:x + crop_size[0], y:y + crop_size[1]]

            # Resize the crop to the input size of the model
            crop = cv2.resize(crop, (self.model.input_shape[1], self.model.input_shape[2]))

            # Normalize the crop
            crop = (crop / 255.0) - 0.5

            # Add the batch dimension
            crop = np.expand_dims(crop, axis=0)
            crop = crop.reshape(1, self.step, self.step, self.num_classes)

            # Perform inference on the crop
            prediction = self.model.predict(crop)[0]

            # Update the prediction map with the prediction for this crop
            prediction = prediction.reshape(self.step, self.step)
            prediction_map[x:x + crop_size[0], y:y + crop_size[1]] = prediction

            # Update the unpredicted mask
            unpredicted_mask[x:x + crop_size[0], y:y + crop_size[1]] = False
            i += 1

        # Normalize the prediction map
        prediction_map = (prediction_map - prediction_map.min()) / (prediction_map.max() - prediction_map.min())

        return prediction_map

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

