import wandb
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class WandbValidationVisualization(Callback):
    def __init__(self, val_data, val_labels, frequency=1, num_images=3):
        super().__init__()
        self.val_data = val_data
        self.val_labels = val_labels
        self.frequency = frequency
        self.num_images = num_images

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch % self.frequency == 0:  # Every 'frequency' epochs
            # Run predictions on the validation data

            random_indices = np.random.choice(len(self.val_data), self.num_images, replace=False)
            val_data_subset = np.take(self.val_data, random_indices, axis=0)
            val_labels_subset = np.take(self.val_labels, random_indices, axis=0)
            with tf.device('/cpu:0'):
                preds = self.model.predict(val_data_subset)

            # Log images to wandb
            for i in range(len(val_data_subset)):
                fig, ax = plt.subplots(1, 3, figsize=(9, 3))
                ax[0].imshow(val_data_subset[i].squeeze(), cmap='gray')
                ax[0].title.set_text('Input Image')
                ax[0].axis('off')

                ax[1].imshow(val_labels_subset[i].squeeze(), cmap='gray')
                ax[1].title.set_text('True Label')
                ax[1].axis('off')

                ax[2].imshow(preds[i].squeeze(), cmap='gray')
                ax[2].title.set_text('Prediction')
                ax[2].axis('off')

                # Log the plot to wandb
                wandb.log({f"Validation Prediction {i}": wandb.Image(plt)})
                plt.close(fig)

# # Usage
# val_viz_callback = WandbValidationVisualization(val_images, val_labels, frequency=10)
#
# # Then pass it to the fit function
# model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), callbacks=[val_viz_callback, ...other callbacks...])
